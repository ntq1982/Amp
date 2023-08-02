TMPDIR := $(shell mktemp -d)
TESTDIR := $(shell cd ../tests; pwd)

main:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90 model/chargeneuralnetwork.f90

tests:
	mkdir -p $(TMPDIR)/amptests
	cd $(TMPDIR)/amptests && nosetests -v --with-timer $(TESTDIR)
