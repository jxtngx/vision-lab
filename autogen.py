import os
import shutil
from pathlib import Path

from keras_autodoc import DocumentationGenerator

rootdir = Path(__file__).parent
docsdir = os.path.join(rootdir, "docs-src", "docs")
srcdocsdir = os.path.join(docsdir, "visionpod")

pages = {"core.PodModule.md": ["visionpod.core.module.PodModule"]}

level = ".." if os.getcwd() == "docs-src" else "."
doc_generator = DocumentationGenerator(pages)
doc_generator.generate(f"{level}/docs-src/docs/visionpod")

root_readme = os.path.join(rootdir, "README.md")
docs_intro = os.path.join(docsdir, "intro.md")
shutil.copyfile(src=root_readme, dst=docs_intro)
