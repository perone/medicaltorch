Rationale & Philosophy
===============================================================================
There are nowadays a lot of repositories with code for parsing medical imaging
data in PyTorch, however, some challenges are still present:

* many of these repositories doesn't contain a single word of documentation;
* many are just for classification datasets;
* the majority of them aren't maintained;
* most of them takes your freedom due to design mistakes;
* many models in these repositories are locked in monolithic code where
  you cannot repurpose for your own goals;
* many of them doesn't contain a single line of testing;
* missing examples on how to use them.

The idea of this framework is to provide a elegant design to solve the issues
of medical imaging in PyTorch. The design principles of this framework are
the following:

* easy reusable components;
* code well-documented;
* documentation with examples and manuals;
* extensive testing coverage;
* easy to integrate in your pipeline;
* support for a variety of medical imaging sources;
* close as possible to the PyTorch design.

With that in mind, there is a long road ahead and contributions are
always welcome.


