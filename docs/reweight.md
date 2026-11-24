
#### Reweighting
During normal compliation denox considers many candidate implementations 
and selects the best dispatch schedule based on profiling results (PGO).
However PGO is slow; it requires compiling thousands of shaders 
and benchmarking them on the GPU. 

One solution around this is to use profiling databases. 
When a database is provided during compilation, it is used as a cache 
for compiled shaders and their benchmarking results. 
The database cache is keyed on the underlying shader source, 
it's parameters such as channel count, assumed input / output size, 
and the device which was used to collect the profiling data.
Importantly it is not keyed based on the weights so recompiling the same
model with different names will be significantly faster as all 
profiling information already exists within a database. 
However databases can quickly grow quite large, as they store all 
candidate spirv code, metadata and profiling information directly 
within a sqlite3 database. 
Databases are the recommended solution to speedup local development, but 
because of their storage size, they are not recommended to be used for 
distributing models.

The `reweight` command tries to address long compiletime 
from a different angle. Given a dnx artefact and a onnx model, 
where the dnx artefact was compiled from a onnx model, with the exact
same topology including input and output names, but with different weights 
(i.e. only onnx initializer data differs).
The reweight command can be used to extract the weights from the 
new onnx model, transform them into their appropriate format 
and update the weights within the dnx artefact directly.
Because this simply replaces the weights it does not require 
compiling and profiling shaders, making reweighting 
the fastest solution to quickly update weights.

But reweighting also has it's limitation, it requires 
that we call reweight with the **same version of denox** that was used to 
compile the input dnx artefact! However it is possible to reweight a model, 
which was originally compiled with the same version but on a different device. 

Initially this seems like a odd requirement, but it becomes 
clearer once we understand what reweight has to do internally. 
A good analogy to think about is a replacing the ro-section of 
C++ executable, which makes heavy use of constant evaluation.
What makes this difficult is that the convertion from compiletime 
inputs to what actually ends up beeing stored 
in the artefact is nontrivial. 
We run into the same problem with reweight. 
ONNX stores all weights in a canonical layout, where the dnx artefacts
store an arbitrarily complex transformation of those weights. 
This can include, different formats, constant propagation, 
analytical rewrites, nontrivial fusions, and potentially more later.
Storing this mapping explicitly within the artefact would quickly becomes 
infeasible and impossible to maintain properly. 

So how does reweight actually achieve this? Similarly to what we would 
do in the C++ analogy we run the compiler passes until we can reconstruct 
the mappings of compiletime inputs (i.e. the ONNX weights) to what we 
store inside of the ro-section (i.e. dnx initializers).
Dropping the analogy; reweight constructs the complete implementation 
search space (supergraph) and searches it for each dispatch within the 
dnx artefact. At this point constant propagation, formats and fusions are already 
decided and weights are available within their transformed dnx representation. 
However this requires all dispatches within the dnx artefact to be elements 
within the implementation search space. 
This is guaranteed when compiling with the same version of denox, regarless 
of the underlying device, however we cannot / will not make this guarantee across 
different versions of of denox, because it would restrict future development to aggressively.





