At the architectural level, this paper is **not** proposing a brand-new sequence backbone. The backbone stays the same; the authors replace the **residual connection across depth**. Their final large model is the **Kimi Linear** MoE Transformer, and the only architectural change is swapping standard residual accumulation for **Attention Residuals (AttnRes)**. In their largest run, Kimi Linear interleaves **Kimi Delta Attention (KDA)** and **Multi-Head Latent Attention (MLA)** in a **3:1 ratio**, each followed by an **MoE feed-forward layer**. Everything else — model depth, hidden size, expert routing, and MLP structure — is unchanged.  

So the clean mental model is:

**base model = PreNorm MoE Transformer / Kimi Linear**
**new thing = replace “add previous hidden state” with “attend over previous layer outputs”**

That is the whole paper in one sentence. 

## 1. What counts as a “layer” here

The paper’s notation is important because it affects implementation. They define (h_l \in \mathbb{R}^d) as the hidden state entering layer (l), and they explicitly say that in Transformer models they treat **each self-attention module or MLP as an individual layer**. So one Transformer block contributes **two depth steps**:

* pre-attention input (\to) attention module output
* pre-MLP input (\to) MLP or MoE output

That means AttnRes is applied **before every attention sublayer and before every MLP/MoE sublayer**, not once per full Transformer block. This is why the pseudocode has separate “pre-attn” and “pre-MLP” AttnRes calls, and why the 27-block final model is described as having **54 layers**. 

## 2. What standard residuals do, and what AttnRes changes

In a normal PreNorm Transformer, the input to layer (l) is effectively the running sum of the embedding and all previous sublayer outputs:
[
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
]
So every earlier layer contributes with fixed coefficient 1. The paper’s claim is that this is too crude: later layers cannot choose which earlier layers matter most, and hidden-state magnitude grows with depth. 

AttnRes changes only this aggregation step. Instead of summing all previous outputs equally, layer (l) forms a **softmax-weighted mixture over earlier depth sources**:
[
h_l = \alpha_{0\to l} h_1 + \sum_{i=1}^{l-1}\alpha_{i\to l} f_i(h_i)
]
where the (\alpha)’s sum to 1 across depth. In other words, each layer gets to **selectively retrieve earlier layer outputs**. 

## 3. Full AttnRes: the exact mechanism

For **Full AttnRes**, every layer (l) has its own learned vector
[
w_l \in \mathbb{R}^d
]
which the paper calls a **pseudo-query**. This is not computed from the current token state; it is just a learned parameter attached to that layer. The earlier layer outputs act as both keys and values:

* source 0 is the token embedding: (v_0 = h_1)
* source (i \ge 1) is the output of earlier layer (i): (v_i = f_i(h_i))
* key (k_i) is just (v_i)

The score from source (i) into layer (l) is:
[
s_{i\to l} = w_l^\top \mathrm{RMSNorm}(k_i)
]
Then softmax over all earlier sources gives:
[
\alpha_{i\to l} = \frac{\exp(s_{i\to l})}{\sum_{j=0}^{l-1}\exp(s_{j\to l})}
]
Finally the actual input to layer (l) is:
[
h_l = \sum_{i=0}^{l-1}\alpha_{i\to l} v_i
]

That is the whole operator. Implementation-wise, for one token position, you keep all previous layer outputs, compute one scalar logit per source, softmax across the **depth axis**, and take the weighted sum of the source vectors. For a full batch/sequence, each source is shape ([B,T,D]); the softmax runs over the source index, not over tokens or channels. 

## 4. Why RMSNorm is inside the depth attention

The RMSNorm is only on the **keys/sources** before scoring. It is there so large-magnitude earlier layers do not dominate the softmax simply because their norm is bigger. The paper’s ablation says removing this hurts both Full and Block AttnRes, and it matters even more in the block version because block summaries can have very different magnitudes. 

So, concretely, the AttnRes operator needs these learnable pieces per sublayer:

* one pseudo-query vector (w_l \in \mathbb{R}^D)
* one RMSNorm over source tensors before taking dot products

The paper says AttnRes adds only **one RMSNorm and one pseudo-query vector per layer**. 

## 5. Why they do not use an input-dependent query in the main model

They tested making the query depend on the current hidden state through a projection, and it worked slightly better in their ablation. But they did **not** use that in the main design because it adds a (D \times D) projection per layer and, more importantly, breaks a major systems advantage: with fixed pseudo-queries (w_l), all queries inside a block can be prepared in parallel without waiting for the sequential forward pass. 

So if you want to match the paper’s main architecture, do **not** make the depth query depend on the current hidden state. Use one learned vector per sublayer, initialized to zero. Zero init is important: the paper says all pseudo-queries must start at zero so the initial attention over sources is uniform, making AttnRes behave like equal-weight averaging at the start of training and avoiding instability. 

## 6. Block AttnRes: the practical version they actually scale

Full AttnRes requires keeping every earlier layer output alive. That is fine conceptually, but expensive in large distributed training. So the scalable architecture is **Block AttnRes**. Layers are partitioned into (N) blocks of (S=L/N) layers each. Within each block, they do plain additive accumulation; across blocks, they do depth attention. 

Define block (n) as a set of (S) consecutive layers (B_n). Its completed block representation is:
[
b_n = \sum_{j\in B_n} f_j(h_j)
]
and the partial sum after the first (i) layers of that block is:
[
b_n^i
]
So Block AttnRes compresses an entire block into a single vector per token, rather than storing every individual sublayer output forever. 

## 7. What each layer in Block AttnRes sees

There is a special source
[
b_0 = h_1
]
which is the token embedding and is always available. Then for a layer inside block (n):

* if it is the **first** layer in the block, it attends over the completed earlier block representations:
  [
  [b_0, b_1, \dots, b_{n-1}]
  ]
* if it is a **later** layer in the same block, it attends over those plus the current block’s running partial sum:
  [
  [b_0, b_1, \dots, b_{n-1}, b_n^{i-1}]
  ]

The same scoring rule is used as in Full AttnRes: dot product between the layer’s pseudo-query and RMS-normalized sources, then softmax over the source list, then weighted sum of the unnormalized source tensors. 

This gives you a very concrete implementation picture:

* **history across completed blocks** = list of block tensors
* **history inside the current block** = one running partial-sum tensor
* **before each sublayer** = run AttnRes over those sources to produce the sublayer input
* **after the sublayer** = add the sublayer output into the running partial sum
* **at block boundary** = freeze the partial sum as a completed block representation and append it to history 

## 8. How one Transformer block executes

Because they treat attention and MLP as separate layers, a Transformer block runs like this.

### Pre-attention step

Take the currently available depth sources and apply AttnRes using the **attention-side pseudo-query** for this sublayer. That produces the input to the attention module. Then apply the normal pre-attention norm and the attention module itself. In the large model, that attention module is either **KDA** or **MLA**, depending on position in the 3:1 interleaving pattern. Add that attention output to the current block’s partial sum. 

### Pre-MLP step

Now take the updated sources and apply AttnRes again using the **MLP-side pseudo-query** for this sublayer. That produces the input to the feed-forward module. Then apply the usual MLP norm and the feed-forward. In the large model, this feed-forward is an **MoE feed-forward layer**. Add that MLP output to the block partial sum. 

That is why the pseudocode has two independent AttnRes projections/norms per Transformer block: one before attention and one before MLP. And that is why Figure 8 separately shows learned depth weights for **pre-attention** and **pre-MLP**. 

## 9. What is learned, and what is not

To implement the main model in the paper, the new learnable things are minimal:

* one pseudo-query vector (w_l) per sublayer
* one RMSNorm for the depth-attention keys per sublayer

Everything else stays from the base model. The depth-attention is **single-head** across depth in the main design. They tried a multihead depth-attention variant and it was worse. They also tried replacing softmax with sigmoid and replacing input-dependent mixing with static mixing; both were worse. So the exact recipe they favor is:

* learned but **input-independent** pseudo-query
* **softmax** over depth sources
* **RMSNorm** on keys
* **single-head** depth mixing across all channels at once 

## 10. Why the softmax is over depth, not sequence

This is easy to miss. The attention inside KDA or MLA is still doing sequence mixing as usual. AttnRes is a **different attention mechanism** whose axis is **depth**. For each token position independently, it chooses how much to use the embedding, earlier sublayer outputs, earlier block summaries, or the current block partial sum. So AttnRes is orthogonal to token-wise attention. It changes how information is mixed **across layers**, not across tokens. 

## 11. The scalable inference design

The architecture itself is Block AttnRes, but the paper also gives a specific implementation strategy so it stays cheap at inference.

Inside one block, there are (S) pseudo-queries (w_l), one per sublayer. Because those queries are learned parameters rather than hidden-state-dependent, you can batch the cross-block attention for all (S) layers in the block.

So they split computation into two phases.

### Phase 1: inter-block attention, batched

For all sublayers in the current block, compute their attention over completed previous block representations in one batched matmul. This gives:

* the inter-block weighted sums
* softmax statistics needed for exact merging later

### Phase 2: intra-block attention, sequential

Walk through the sublayers one by one, updating the current block’s partial sum. Each sublayer additionally attends to the current block partial sum. Then combine Phase 1 and Phase 2 contributions using **online softmax**, so the result is exactly the same as if you had attended over the union of all sources at once. 

This is implementation-important because it tells you how to make Block AttnRes efficient without changing its math. 

## 12. Training-time distributed systems design

For large-scale training, the main problem is pipeline communication. A naïve distributed implementation would resend the entire accumulated block history at every stage boundary. Their solution is **cross-stage caching**: each rank caches block representations received earlier, and later stages send only the incremental new blocks. They report this reduces communication from scaling with total chunks to scaling with physical pipeline stages, and keeps end-to-end training overhead below 4% under pipeline parallelism. 

For long-context inference, they also shard stored block representations across tensor-parallel devices along sequence length, and merge the online-softmax part into the normal TP communication path. That is a systems optimization, not a change to the model equations. 

## 13. The exact large-model configuration

Their final large model is:

* **Kimi Linear**
* **27 Transformer blocks = 54 sublayers**
* each Transformer block has attention then MoE FFN
* attention types interleaved **KDA : MLA = 3 : 1**
* MoE has **8 routed experts out of 256**, plus **1 shared expert**
* total parameters **48B**, activated parameters **3B**
* Block AttnRes uses **6 layers per block**, so 54 layers become **9 AttnRes blocks**, and with the token embedding included as source 0 there are **10 depth-wise sources** visible to the top of the network 

That is the exact model they train at scale in this paper. 

## 14. What you can implement from this paper alone

You can implement the **AttnRes mechanism itself** from this paper alone, because the equations and pseudocode are sufficient:

* define the source tensors,
* define the pseudo-query per sublayer,
* RMSNorm the sources for scoring,
* softmax over source index,
* weighted sum of source tensors,
* update current block partial sum,
* freeze it at block boundaries,
* batch inter-block attention inside blocks if you want the optimized inference path. 

What you **cannot** fully reconstruct from this paper alone is the internal math of **KDA** and **MLA**, because this paper treats them as inherited backbone modules from Kimi Linear and does not re-derive them. So:

* you **can** implement AttnRes on top of any PreNorm Transformer or MoE Transformer from this paper,
* but to reproduce the exact **Kimi Linear + KDA/MLA** backbone, you would also need the Kimi Linear paper. 

## 15. The shortest faithful description

The architecture they use is:

**a Kimi Linear MoE Transformer in which every attention sublayer and every MLP/MoE sublayer receives its input not from a unit-weight residual sum, but from a softmax-weighted retrieval over earlier depth representations.**

In the scalable version, earlier sublayers are compressed into **block summaries**, and each current sublayer attends to:

* the embedding,
* all completed previous block summaries,
* and, if it is not the first sublayer of the block, the current block’s running partial sum.  

That is the architecture.

-------------------------------------------------------------------------------------
The right way to adapt this paper to chess is to treat **AttnRes as the transferable idea**, not the whole Kimi Linear stack.

In the paper, the base model is still a language-model backbone; AttnRes only changes how representations are mixed **across depth**. Concretely, instead of the usual residual sum (h_l = h_1 + \sum_{i<l} f_i(h_i)), each layer forms a softmax-weighted mixture of earlier layer outputs, using a learned query and RMS-normalized source representations. The scalable variant groups layers into blocks and attends over block summaries rather than every individual layer. The paper’s large model keeps the Kimi Linear MoE backbone and only swaps the residual path.  

For chess, that means: **do not think “how do I turn their generator into a regressor?”** Think instead: **“what should the chess backbone be, and where should AttnRes sit inside it?”** The answer is: use a board encoder + bidirectional reasoning stack + scalar value head, and let AttnRes control how each reasoning step retrieves earlier depth states. The token-generation machinery, long-context optimizations, and KDA/MLA choices are not the main point for your setting. The paper itself says the core novelty is the replacement of fixed residual accumulation with depth-wise softmax selection. 

## My recommendation in one sentence

I would build a **bidirectional board Transformer with AttnRes over depth, plus a value token and a tanh scalar head**, and if you want variable compute, I would **tie the main reasoning block across steps** so that harder positions can run for more recurrent steps before producing the final value.

That gives you the two properties you care about:

* a natural fit for **static board evaluation**
* a clean path to **searchless “think longer on hard positions” inference**

## First design choice: what to keep from the paper, and what to change

Keep these parts:

1. **AttnRes itself**: each sublayer should retrieve a weighted mixture of earlier depth states instead of just adding the immediately previous one. That is the core mechanism. 
2. **RMSNorm on the depth-attention keys** before scoring. The paper found it matters. 
3. **Zero-initialize the depth queries** so training starts close to uniform mixing rather than unstable sharp routing. The paper explicitly calls this out. 
4. **Apply depth mixing before both the attention sublayer and the MLP sublayer**, not just once per full block. In their notation, attention and MLP are separate layers. 

Change these parts:

1. Replace **causal language attention** with **bidirectional board attention**.
2. Replace the **token decoder** with a **scalar value head**.
3. Replace long-context sequence engineering with **2D board geometry biases**.
4. Because chess has tiny input length compared with language, you can often use **Full AttnRes** instead of Block AttnRes and skip all the infrastructure tricks. The paper uses Block AttnRes mainly to cut memory and communication at LLM scale. 

And I would probably *not* copy these on the first pass:

1. **KDA / MLA**
2. **MoE**
3. **two-phase inference and pipeline caching**
4. **long-context prefilling machinery**

Those solve LLM scaling problems, not chess-evaluation problems. The paper’s backbone is “identical to Kimi Linear” except for AttnRes, which is a strong hint that those pieces are not the essence of the method. 

## The model I would actually build

I would structure the chess evaluator as five parts:

1. a **board tokenizer**
2. a **stem embedding layer**
3. a **repeated reasoning block**
4. an **AttnRes mixer over depth**
5. a **value head**

Now I’ll describe each one as if you were implementing it from scratch.

---

## 1. Board tokenizer

Do **not** collapse the whole position into one flat vector and feed it to an MLP if your goal is to benefit from the paper’s architecture. AttnRes helps with depth selection, but the actual chess reasoning still needs a tokenized representation where relations can be built by self-attention.

The cleanest representation is:

* **64 square tokens**, one per square
* **1 value token**, whose job is to collect global information and drive the final scalar prediction
* **1 state token**, carrying side to move, castling rights, en passant file, halfmove clock, repetition info, maybe move number if you care

So you have about **66 tokens** total.

Each square token should represent:

* which piece is on that square: 12 piece types plus empty
* the square’s coordinates
* optionally a small encoding of square color or board parity

The state token should represent the non-geometric metadata that is not naturally attached to a single square.

Why this is the right abstraction: in chess, most evaluation depends on **relations between pieces and squares**. A tokenized board lets self-attention build those relations explicitly; a flat vector makes the model rediscover that structure in a less natural way.

## 2. Stem embedding

Each token is projected into a common hidden size (d).

For each square token, the embedding should be the sum of:

* a **piece embedding**
* a **square embedding**
* optionally a **small 2D coordinate embedding** split into file and rank components

The value token and state token each get their own learned base embedding, plus state features added to the state token.

At this point, your hidden state is a tensor of shape:

* number of tokens × hidden dimension

for one position.

This initial tensor is the source the paper would call **(h_1)** or **(b_0)**, depending on whether you use Full or Block AttnRes. In your chess model, this stem output should always remain available as the earliest source in depth mixing, just as the paper always keeps the token embedding available as source 0. 

## 3. The reasoning backbone

Because the input is static and short, I would use a **standard bidirectional Transformer block** as the core compute unit:

* PreNorm
* multi-head self-attention over all board tokens
* feed-forward MLP, preferably SwiGLU or GEGLU style
* no causal mask

No causal mask is important. This is not generation. Every square should see every other square immediately.

And because token count is tiny, I would use **full softmax self-attention**, not linear attention. The long-context efficiency argument from the language model is irrelevant here.

So the core sublayers are ordinary:

* attention sublayer
* MLP sublayer

The paper’s trick is not inside those sublayers. It is in what hidden state you feed into them. 

## 4. Where AttnRes fits

This is the central part.

In the paper, each layer receives not the plain residual sum, but a weighted mixture over earlier depth sources:
[
h_l = \sum_{i<l} \alpha_{i\to l} v_i
]
where the weights come from a softmax over scores between the current layer’s query and the normalized earlier sources. 

For chess, I would use the same idea, but interpret depth as **reasoning steps over the same board**.

There are two good versions.

### Version A: simplest and strongest baseline

Use **untied depth** with **Full AttnRes**.

That means:

* choose a fixed number of sublayers, say 24 or 32
* every attention sublayer and every MLP sublayer can attend over **all previous sublayer outputs**
* no recurrence, no adaptive compute yet

This is the cleanest direct port of the paper.

Because your sequence length is only ~66 tokens, and your total depth is probably a few dozen sublayers, Full AttnRes is completely practical. The paper presents Block AttnRes mainly as a scaling optimization when storing every earlier layer output becomes painful at LLM scale. 

### Version B: the one that matches your “harder positions get more compute” goal

Use a **shared recurrent reasoning block** repeated for (K) steps, and apply AttnRes over the previous step outputs.

This is the version I would recommend for searchless chess.

Each recurrent step contains:

1. one **attention sublayer**
2. one **MLP sublayer**

So one recurrent step is basically one Transformer block.

Now the clever part: treat each completed step as one depth source.

At step (s), the model has available:

* the initial stem representation (b_0)
* previous completed step summaries (b_1, b_2, \dots, b_{s-1})

Before the attention sublayer of step (s), AttnRes mixes those sources to create the input to attention.

After attention produces its output (a_s), you now have a partial within-step representation.

Before the MLP sublayer, AttnRes mixes:

* (b_0, b_1, \dots, b_{s-1})
* plus the current partial representation (a_s)

Then the MLP runs, and the completed step summary becomes:

* (b_s = a_s + m_s)

This is exactly the Block AttnRes pattern from the paper, with **one recurrent step acting like one AttnRes block** whose two internal layers are attention and MLP. The paper’s block formulation is “completed previous blocks plus current partial block”; that maps almost perfectly to this recurrent chess setting. 

That gives you a very clean iterative evaluator:

* each step refines the board understanding
* later steps can explicitly retrieve earlier, more local or tactical representations
* the number of steps becomes your inference-time compute knob

## 5. What the depth-attention query should be

The paper’s main model uses a **learned per-layer pseudo-query** (w_l), independent of the current hidden state. That choice is cheap and parallelizable. They also report an ablation where an **input-dependent query** performs slightly better, but they did not use it in the main large model for systems reasons.  

For chess, I would deviate from the paper here.

Because your model is small and fixed-input, the systems argument mostly disappears. So I would use an **input-dependent depth query**, derived from the current position state.

The cleanest design is:

* for the pre-attention AttnRes mixer, build the query from the current **value token**
* for the pre-MLP AttnRes mixer, again build the query from the updated value token or a pooled summary

Why this helps:

* easy positions and hard positions should not retrieve the same depth sources
* tactical positions may need to look back to earlier sharp local feature maps
* positional positions may prefer smoother global summaries
* an input-dependent query lets the position itself decide what earlier reasoning state is useful

If you want a simpler first version, use the paper’s fixed learned query. But for the chess setting, I think the input-dependent version is more natural.

## 6. How the AttnRes mixer actually operates in the chess model

This part has to be crystal clear, because it is easy to misunderstand.

Each depth source is **not a single vector**. It is the **entire token matrix** for that source:

* value token
* state token
* 64 square tokens

So if you have (S) sources, you are storing (S) tensors of shape:

* tokens × hidden dimension

The AttnRes softmax is computed **over the source axis**, separately for each token position.

That means:

* the current value token decides how much to take from earlier value-token states
* the current e4 token decides how much to take from earlier e4-token states
* and so on

AttnRes itself does **not** do cross-token mixing. Cross-token mixing still happens inside the self-attention sublayer. AttnRes only decides **which depth state each token should start from** before the next computation.

That is exactly how it should be for chess:

* self-attention handles spatial relations between squares and pieces
* AttnRes handles temporal/depth selection between earlier reasoning states

## 7. Why Full AttnRes is probably better than Block AttnRes for your first chess model

The paper introduces Block AttnRes because keeping all earlier layer outputs is painful for huge language models and distributed training. In chess, neither input length nor infrastructure pressure is large. 

So if you want the simplest high-quality implementation, I would start with:

* **Full AttnRes**
* no MoE
* no KDA/MLA
* no two-phase block inference

That way, each sublayer can retrieve any earlier sublayer exactly, which is probably ideal for chess motifs.

Only move to Block AttnRes if:

* you make the network very deep,
* or you insist on a recurrent shared-weight model and want a very clean step-level memory.

## 8. How to add variable inference time

This is the part most aligned with your earlier BDH question.

A standard untied 24-layer network has fixed compute. If you want harder positions to consume more compute, make the compute block **recurrent**.

The structure is:

* one shared attention module
* one shared MLP module
* one shared AttnRes mechanism
* repeated for (K) steps

At each step, the model emits:

* an intermediate value estimate
* optionally a halting/confidence score

Then inference works like this:

1. run 2 or 3 steps no matter what
2. after each subsequent step, check whether the value estimate has stabilized
3. stop when it has stabilized, or when the halting head says “enough”

A simple stopping criterion is:

* small change in scalar value between consecutive steps
* plus maybe low entropy or high confidence in a secondary WDL head if you add one

So easy positions stop early.
Hard positions run longer.
No search tree is involved.

This is the cleanest way to turn AttnRes into a searchless adaptive-compute evaluator.

## 9. What the value head should look like

The value head should read the **final value token** after the last reasoning step.

Do not pool all square tokens equally unless you have a strong reason. A dedicated value token is cleaner because:

* it has a single place to accumulate global evaluation state
* it gives the model an explicit “evaluation workspace”
* it makes intermediate value supervision easy at every step

So the head is:

* take the value token
* apply a small MLP
* project to one scalar
* apply `tanh`

That gives the output in ([-1,1]), matching your target convention.

I would also consider an auxiliary **WDL head** from the same value token:

* win probability
* draw probability
* loss probability

Even if your primary output is scalar value, the WDL head often stabilizes learning because it teaches coarse game-state semantics. But it is optional.

## 10. Board geometry inside token attention

AttnRes is only half the architecture. The token attention still needs chess-specific geometry.

So inside the self-attention sublayer, I would add **2D relative position information**. The model should know:

* file distance
* rank distance
* maybe diagonal relation
* maybe knight-offset relation as a special learned bias if you want to be fancy

This is important because chess relations are geometric. A queen on d1 and a king on h5 matter partly because of the line connecting them, not just because those two token identities co-occur.

You do not need anything exotic. Simple learned 2D relative biases are enough to start.

## 11. Whether to use MoE

The paper’s large backbone is MoE because it is an LLM-scale design. 

For chess value regression, I would not start with MoE.

A dense MLP is the right first version. MoE adds routing complexity and makes it harder to understand whether improvements come from:

* better depth mixing
* or expert specialization

Once the dense model works, MoE can become attractive because experts might naturally specialize into:

* tactical positions
* endgames
* closed structures
* king attacks
* opposite-side castling
* fortress-like positions

But I would treat that as a second-stage scale-up, not part of the base adaptation.

## 12. The exact forward pass I would recommend

Here is the model flow in words.

### Step 0: create the initial source

Encode the board into 64 square tokens, one state token, and one value token.
Project them to hidden dimension.
This token matrix is source (b_0).

### Step 1: first reasoning step

Use AttnRes to mix the currently available depth sources for the attention sublayer.
At the first step, that is just (b_0), so nothing fancy happens yet.
Run bidirectional self-attention.
Call its output (a_1).

Now run AttnRes again before the MLP, now with:

* (b_0)
* plus the partial within-step representation (a_1)

Run the MLP and get (m_1).

Define the completed step summary:

* (b_1 = a_1 + m_1)

Now predict an intermediate value from the value token inside (b_1).

### Step 2 and later

For step (s), the attention sublayer receives an AttnRes mixture over:

* (b_0, b_1, \dots, b_{s-1})

Then the MLP sublayer receives an AttnRes mixture over:

* (b_0, b_1, \dots, b_{s-1}, a_s)

Then you form:

* (b_s = a_s + m_s)

Again read out the current value.

Repeat until:

* you hit the max step budget,
* or the estimate stabilizes.

That is the entire inference algorithm.

## 13. The design I would choose first

If your goal is strongest conceptual fit for searchless chess, I would start with this:

**Chess AttnRes Recurrent Evaluator**

* 64 square tokens + state token + value token
* dense bidirectional self-attention
* dense MLP
* Full AttnRes or step-level Block AttnRes
* shared block weights across recurrent steps
* input-dependent depth query from the value token
* scalar tanh value head
* intermediate value prediction at every step
* adaptive stopping at inference

That is the model I think best matches your use case.

## 14. The one practical caveat

AttnRes by itself improves **how depth states are selected**. It does not automatically solve the “spend more compute on harder positions” problem. For that, you still need one of these:

* shared recurrent block weights
* a step budget (K_{\max})
* intermediate predictions
* a stopping rule or halting head

So in chess, the real adaptation is:

**AttnRes + recurrent shared block + scalar head**

not just

**AttnRes + ordinary fixed-depth transformer**.

The latter is still useful, but it does not give you adaptive thinking time.

## Bottom line

For chess, I would not literally port the paper’s full language-model stack.

I would port the **depth-attention residual mechanism**, place it inside a **bidirectional board transformer**, and reinterpret depth as **iterative reasoning over one fixed position**. The model should read a tokenized board, repeatedly refine it through shared attention/MLP computation, use AttnRes to retrieve the most relevant earlier reasoning states, and emit a scalar value from a dedicated value token. The paper supports the core mechanism — depth-wise softmax retrieval with RMSNorm and per-sublayer application — while the recurrent adaptive-compute chess setup is the natural extension for your problem.  

I can next turn this into a concrete layer-by-layer spec with tensor shapes and exact dataflow for one forward pass.
