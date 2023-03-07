import os
import shutil
from pathlib import Path

from keras_autodoc import DocumentationGenerator

FILEPATH = Path(__file__)
PKGPATH = FILEPATH.parents[2]
PROJECTPATH = FILEPATH.parents[3]
DOCSDIR = os.path.join(PROJECTPATH, "docs-src", "docs")


class PodDocsGenerator:
    def build():

        pages = {
            "core.PodModule.md": [
                # "visionpod.core.module.ViT_B_16_Parameters",
                # "visionpod.core.module.ViT_B_16_HyperParameters",
                "visionpod.core.module.PodModule",
            ]
        }

        doc_generator = DocumentationGenerator(pages)
        doc_generator.generate(f"{PROJECTPATH}/docs-src/docs/visionpod")

        root_readme = os.path.join(PROJECTPATH, "README.md")
        docs_intro = os.path.join(DOCSDIR, "intro.md")

        shutil.copyfile(src=root_readme, dst=docs_intro)

        with open(docs_intro, "r", encoding="utf-8") as intro_file:
            text = intro_file.readlines()

        with open(docs_intro, "w", encoding="utf-8") as intro_file:
            text.insert(0, "---\n")
            text.insert(1, "sidebar_position: 1\n")
            text.insert(2, "---\n\n")
            intro_file.writelines("".join(text))

        intro_file.close()
