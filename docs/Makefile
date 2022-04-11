TUT:=$(shell basename $(CURDIR))

default: $(TARGETS) Slides

Slides: $(TUT).md
	pandoc --filter mermaid-filter --highlight-style pygments -t beamer -V theme:metropolis -V aspectratio:169 $(TUT).md -o $(TUT)_$@.pdf
