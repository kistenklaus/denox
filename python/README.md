# Evaluation data pipeline

With denox we can populate a database, benchmark it and finally export it as a csv file.
These csv files are gigantic, and not really usable because that would take days.
That's why we first preprocess them, here we aggregate sample of the same implementation. 
That will take a lot of time, because it's like 30 million samples or something crazy, so 
be patient.

During preprocessing we also split the data by shader implementaiton. 
The CSV stores things like input / output dimensions and shader parameters as strings.
After having the samples aggregates, we split all data by implementation and 
then parse those strings. The result of this is written as parquet files to disk.

When i say split here, i mean group by "shader", "operation", the problem is that the 
"operation" string can be quite large, and using is as a filename doesn't really work because it can contain 
special characters and so on, so we hash the operation. It's not the best but essentially 
after preprocessing we end up with something like:
```
concat-conv-cm-020cc91d46eb2a85.parquet # leaky_relu(conv2d(concat([x,y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)),alpha=0.001)
concat-conv-cm-104397bc5fc26829.parquet # conv2d(concat([x,y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1))
concat-conv-cm-81130abaff94c27f.parquet # relu(conv2d(concat([x,y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)))
concat-conv-cm-ad3becf79c230fc1.parquet # relu(conv2d(concat([upsample(x,mode=nearest,scaling_factor=2),y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)))
concat-conv-cm-e72ad629fd314864.parquet # conv2d(concat([upsample(x,mode=nearest,scaling_factor=2),y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1))
```
All of them have the same underlying GLSL implementation, but all implement different operations. 
To find out it's easiest to just load them, and check `df["operation"].unique()`

Because preprocessing takes ages, here is a google drive link with all of the parquet files,
just place them inside the `python/parquet/` directory. (TODO)


# General workflow i use it to find policies.
As a example we can look at `direct_conv_cm.py`, we first load the 
parquet files, that are called `direct-conv-cm-*`, which corresponds to 
all samples of the `direct-conv-cm` GLSL shader template, for but different logical 
operations, and for different input / output shapes, layouts etc.

We then first do some filtering to select for example only a single operation, like conv+relu or example.

Afterwards we derive some values specific to this shader, for example for a direct-conv-cm shader,
because we know the config parameters, we derive things like shared memory, tile sizes or
bad estimates for registers (just count the amount of coopmats used at the peek, and then derive how many 
u32 registers those would correspond to, i know that's not at all what the hardware does, that why i say BAD).

Now we define policies. For example:
```python
def shared_memory_policy(df: pd.DataFrame) -> pd.DataFrame:
    # intentionally unrestrictive right now.
    MIN_SHARED_MEMORY = 1024  # 1KiB
    MAX_SHARED_MEMORY = 102400  # 100KiB
    mask = df["sh_size"].between(MIN_SHARED_MEMORY, MAX_SHARED_MEMORY)
    return df.loc[mask]
```
Policies filter the possible configurations, based on some rule. Here 
we filter by shared memory. For now all of these policies are platform independent,
so they are generally not very restrictive. In this example above 100KiB is a unreasonable 
about of shared memory regardless of the vendor (AMD, Intel, NVIDIA).


After defining a bunch of policies we apply the once that we what to see, 
when we want to try turn on / off policies we simply comment / uncomment the 
policy here.

Finally we plot a histogram of the relative speedup, 
which is computed during preprocessing for each shader relative to the best config which executed the exact same logical operation.
We also plot the histogram over the original full search space. 

Then it's just trail and error to find good policies, and at the end we hardcode those policies in 
our compiler. 

Lastly a quick note on the datasets, generally their are quite, big but still they don't represent all compilable configurations,
because that would probably be terabytes of SPIR-V shaders, instead some rules are already applied even with this large 
dataset for example for the conv implementations the `perfect_ktiling_policy` is already implemented, because otherwise the 
search space would blow up completely, similarly shared memory and registers are limited to some extend, but always very conservative,
like less than 200 registers per invocation or less than 0.75 * maxComputeSharedMemory shared memory.

How do we pick reasonable configs, generally we don't expect the best configurations to survive all 
policies, we are fine with policies which do remove absolutely best implemnetations as long as they 
remove a significant amount of really slow configs; We can still use those policies at lower optimization levels, 
where we would want a very small search space and are fine with missing the fastest configuration, if it means that 
compilation + benchmarking takes 5min compared to 5h. 



### Top-K search space reduction:
The idea is the following as we measurements from a large set of GPUs, with different architectures, 
we can preprocess your search space to find configurations never competitive. 

We take our measurements, and group them by (logical-operation, device), now we look at 
best K performing configurations this group. Afterwards we union all of the top k configurations of
all groups together and remove duplicates, we now have a list of configurations, which are at least good 
for one (operation, device) in our dataset. Or in other words, a configuration was never in the top k,
is not included. 


### Set Cover Idea:
Top-K is already actually quite good, it ensures that configs, which are never best are not picked, the 
problem with top-k is that it only selects based on the rank, and if we have multiple configurations, which 
almost perform identical, it should not matter, which one of those we pick.
What we actually want to do is ensure that for all (device, operation) groups, 
we take a least G many configurations, which are good (i.e. have a relative-speedup $s_{rel} > \alpha$.
A secondary goal is to reduce the amount of candidate 
configurations. So if one configurations only perform well on a single logical operation, but performs worse 
everywhere else, but another configuration exists which is generally good across multiple devices and logical 
operations, we should prefer it over the other. 
This is a classic set cover minimization problem. \
Let $\Phi$ be the set of (device, logical-operation) groups. 
And $\Sigma$ be the set of all configurations. \
**Set-Cover Instance:**
Let the $U = \Phi$ be the universe and the 
subsets $S_c = \left\{ t \in \Phi\ \vert\ s_{rel}(t, c) > \alpha \right\} \forall c \in \Sigma$ \
Solving this instance, yields a list of configuraitons $\Sigma'$, where for 
$\forall t \in \Phi:\exists c \in \Sigma': s_{rel}(t,c) > \alpha$.
This is already good, but we actually want something a bit less strict, because picking just a single 
good configuration per group is dangerous, as it's very unclear if the picked instance will also 
perform well on other architectures or for operations not the dataset. 
We can mitigate this my solving a multiset cover problem, instead where each $t \in T$ has to be 
covered by at least $G$ many configurations.\
*We implement this with ILP solvers*.\
*It could also be interessting to consider weighted multiset cover, where we prefer configurations with higher relative-speedup.*


### More ILP Ideas

<!-- $$ -->
<!-- \min \beta  \sum_{t\in \Phi} \frac{1}{\Sigma_t} \sum_{c \in \Sigma_t} L_{t,c}x_c  -->
<!-- + \sum_{t \in \Phi} \sum_{c \in \Sigma_t} (L_{t,c} - L_t^*)y_{t,c} -->
<!-- $$ -->
<!--  -->
<!--  -->
<!-- $$ -->
<!-- \min \beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t} L_{t,c}x_c  -->
<!-- + \sum_{t \in \Phi} \sum_{c \in \Sigma_t} (L_{t,c} - L_t^*)y_{t,c} -->
<!-- $$ -->
<!--  -->
<!--  -->

<!-- $$ -->
<!-- \min \left\{  -->
<!-- \beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t} L_{t,c}x_c  -->
<!-- + \sum_{t \in \Phi} \min_{c \in \Sigma_t, x_c = 1} \left\{L_{t,c} - L_t^* \right\} -->
<!-- \right\} -->
<!-- $$ -->
<!--  -->
<!--  -->
<!-- $$ -->
<!-- \argmin_{\Sigma' \subseteq \Sigma} \left\{  -->
<!--  -->
<!-- \beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t \cup \Sigma'} L_{t,c} -->
<!-- +  -->
<!-- \alpha \sum_{t \in \Phi} \min_{c \in \Sigma_t \cup \Sigma'} \left\{L_{t,c} - L_t^* \right\} -->
<!-- \right\} -->
<!-- $$ -->

Minimization problem:
$$
\argmin_{\Sigma' \subseteq \Sigma} \left\{ 

\beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t \cup \Sigma'} L_{t,c}
+ 
\alpha \sum_{t \in \Phi} \sum_{i=1}^{n_t} \lambda_i (L_{t,(i)} - L_t^*)

\right\}
$$

Where $L_{t,(i)}$ represents the i-th 
best config $c \in \Sigma_t \cup \Sigma'$ in group $t$\
and $n_t = \min(G, \vert\Sigma_t\vert)$ \
$\lambda_i$'s are weights, maybe something like the harmonic series
$$
\lambda_i = \frac{1/i}{\sum_{j=1}^{n_t} 1/j}
$$
or just uniform, let's see.

###### MILP:

$$
\min \left\{ 
\beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t} L_{t,c}x_c 
+ 
\alpha \sum_{t\in \Phi} \sum_{i=1}^{n_t} \sum_{c \in \Sigma_t} \lambda_i (L_{t,c} - L_t^*) y_{t,c,i}
\right\}
$$
$$
\forall t : \forall c : z_{t,c} \le x_c \land x_c, z_{t,c} \in \left\{0,1\right\}
$$
$$
\sum_{c \in \Sigma_t} y_{t,c,i} = 1\ \ \forall i
$$

In total $\sum_{t \in \Phi} G \vert \Sigma_t \vert$ many variables.\

With uniform weights $\lambda_i = \frac{1}{n_t}$ this becomes much more tracable:
$$
\min \left\{ 
\beta  \sum_{t\in \Phi} \sum_{c \in \Sigma_t} L_{t,c}x_c 
+ 
\alpha \sum_{t \in \Phi} \sum_{c \in \Sigma_t} \frac{1}{n_t}(L_{t,c} - L_t^*) y_{t,c}
\right\}
$$
$$
\forall t : \forall c : z_{t,c} \le x_c \land x_c, z_{t,c} \in \left\{0,1\right\}
$$
$$
\sum_{c \in \Sigma_t} y_{t,c} = n_t
$$
Now with only $\sum_{t \in \Phi} \vert \Sigma_t \vert$ variables, (still a couple thousand).





