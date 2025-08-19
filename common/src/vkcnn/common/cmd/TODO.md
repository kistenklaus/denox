TODO: Refactor this: it's probably significantly better
to have a shader registry because spirv source sections
might be duplicated.

Also we probably should fine a different way to encode debug information,
possibly with a optional file section or something like this. But then how do
we store the offsets.

TODO: It would also be nice to have a offset table for random access.

So basically something like a big header which gives offsets for the individual sections.
Then each section starts with a offset table, followed by it's data.

The only downside here is that we would now have to linearlize the memory explicitly but that's probably fine.

