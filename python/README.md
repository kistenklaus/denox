# Evaluation data pipeline

With denox we can populate a database, benchmark it and finally export it as a csv file.
These csv files are gigantic, and not really usable as is because that would take days.
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
just place them inside the python/parquet/ directory.









