# content of docs/autogen.py

from keras_autodoc import DocumentationGenerator

pages = {"core.PodModule.md": ["visionpod.core.module.PodModule"]}

doc_generator = DocumentationGenerator(pages)
doc_generator.generate("./docs-src/docs/visionpod")