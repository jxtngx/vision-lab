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
                "visionpod.core.module.PodModule",
                "visionpod.core.module.PodModule.forward",
                "visionpod.core.module.PodModule.training_step",
                "visionpod.core.module.PodModule.test_step",
                "visionpod.core.module.PodModule.validation_step",
                "visionpod.core.module.PodModule.predict_step",
                "visionpod.core.module.PodModule.configure_optimizers",
                "visionpod.core.module.PodModule.common_step",
            ],
            "core.PodTrainer.md": [
                "visionpod.core.trainer.PodTrainer",
                "visionpod.core.trainer.PodTrainer.persist_predictions",
            ],
            "app.AutoScaledFastAPI.md": ["visionpod.app.app.AutoScaledFastAPI"],
        }

        doc_generator = DocumentationGenerator(pages)
        doc_generator.generate(f"{PROJECTPATH}/docs-src/docs/VisionPod")

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
