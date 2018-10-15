Rationale & Philosophy
===============================================================================
Nowadays there are a lot of repositories with code for parsing medical imaging
data in PyTorch, however, some challenges still remain:

* many of these repositories don't contain a single word of documentation;
* many are just for classification datasets;
* the majority of them aren't maintained;
* most of them take your freedom due to design mistakes;
* many models in these repositories are locked in monolithic code where
  you cannot repurpose for your own goals;
* many of them don't contain a single line of testing;
* missing examples on how to use them.

The idea of this framework is to provide an elegant design to solve the issues
of medical imaging in PyTorch. The design principles of this framework are
the following:

* easy reusable components;
* well-documented code and APIs;
* documentation with examples and manuals;
* extensive testing coverage;
* easy to integrate into your pipeline;
* support for a variety of medical imaging sources;
* close as possible to the PyTorch design.

With that in mind, there is a long road ahead and contributions are
always welcome.
