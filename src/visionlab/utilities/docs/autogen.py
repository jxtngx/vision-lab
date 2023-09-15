import os
import shutil
from pathlib import Path

from keras_autodoc import DocumentationGenerator

FILEPATH = Path(__file__)
package = FILEPATH.parents[2]
project = FILEPATH.parents[4]
DOCSDIR = os.path.join(project, "docs-src", "docs")


class LabDocsGenerator:
    def build():
        pages = {
            "core.LabModule.md": [
                "visionlab.core.module.LabModule",
                "visionlab.core.module.LabModule.forward",
                "visionlab.core.module.LabModule.training_step",
                "visionlab.core.module.LabModule.test_step",
                "visionlab.core.module.LabModule.validation_step",
                "visionlab.core.module.LabModule.predict_step",
                "visionlab.core.module.LabModule.configure_optimizers",
                "visionlab.core.module.LabModule.common_step",
            ],
            "core.LabTrainer.md": [
                "visionlab.core.trainer.LabTrainer",
                "visionlab.core.trainer.LabTrainer.persist_predictions",
            ],
            "app.AutoScaledFastAPI.md": ["visionlab.app.app.AutoScaledFastAPI"],
        }

        doc_generator = DocumentationGenerator(pages)
        doc_generator.generate(f"{project}/docs-src/docs/Source Reference")

        root_readme = os.path.join(project, "README.md")
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
