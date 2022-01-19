import os
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import re
from enum import Enum

import fire
from sklearn.metrics import classification_report, confusion_matrix

O_TOKEN = "O_TOKEN"
TASK_CONCEPT = "CONCEPT"
TASK_ASSERTION = "ASSERTION"
NB_PRED = "nb_predictions"
NB_GROUND_TRUTH = "nb_ground_truth"
TP = "tp"
RECALL = "recall"
PRECISION = "precision"
F1_SCORE = "f1_score"


@dataclass
class Token:
    label: str
    text: str
    line: int
    word_index: int


@dataclass
class EntityAnnotation:
    label: str
    text: str
    start_line: int
    end_line: int
    start_word: int
    end_word: int


class RelationValue(Enum):
    TrAP = "TrAP"
    TrNAP = "TrNAP"
    TrCP = "TrCP"
    TeRP = "TeRP"
    TeCP = "TeCP"
    TrIP = "TrIP"
    PIP = "PIP"
    TrWP = "TrWP"


@dataclass
class EntityAnnotationForRelation:
    text: str
    start_line: int
    end_line: int
    start_word: int
    end_word: int


@dataclass
class RelationAnnotation:
    label: RelationValue
    left_entity: EntityAnnotationForRelation
    right_entity: EntityAnnotationForRelation


class Evaluator:
    def __init__(self):
        self.concept_annotation_dir = None
        self.concept_prediction_dir = None
        self.assertion_annotation_dir = None
        self.assertion_prediction_dir = None
        self.relation_annotation_dir = None
        self.relation_prediction_dir = None
        self.entries_dir = None

    def evaluate(self, concept_annotation_dir: str, concept_prediction_dir: str,
                 assertion_annotation_dir: str, assertion_prediction_dir: str, relation_annotation_dir: str,
                 relation_prediction_dir: str, entries_dir: str):

        self.concept_annotation_dir = concept_annotation_dir
        self.concept_prediction_dir = concept_prediction_dir
        self.assertion_annotation_dir = assertion_annotation_dir
        self.assertion_prediction_dir = assertion_prediction_dir
        self.relation_annotation_dir = relation_annotation_dir
        self.relation_prediction_dir = relation_prediction_dir
        self.entries_dir = entries_dir

        print("############# CONCEPT EVALUATION #############")
        f1_concept = self.evaluate_concept()
        print("\n\n############# ASSERTION EVALUATION #############")
        f1_assertion = self.evaluate_assertion()
        print("\n\n############# RELATION EVALUATION #############")
        f1_rel = self.evaluate_relation()
        global_score = (f1_concept + f1_assertion + f1_rel) / 3
        print("\n\n############# GLOBAL SCORE #############")
        print(round(global_score, 2))

    def evaluate_concept(self) -> float:
        y_true, y_pred = self._load_entities(TASK_CONCEPT)
        sorted_labels = [label for label, _ in sorted(Counter(y_true).items(), key=lambda c: (c[0] != O_TOKEN, -c[1]))]
        print("## Confusion matrix for concepts ##")
        print(confusion_matrix(y_true, y_pred, labels=sorted_labels))
        print("\n## Classification report for concepts ##")
        print(classification_report(y_true, y_pred, labels=sorted_labels))
        class_report = classification_report(y_true, y_pred, labels=sorted_labels, output_dict=True)
        return class_report["macro avg"]["f1-score"]

    def evaluate_assertion(self) -> float:
        y_true, y_pred = self._load_entities(TASK_ASSERTION)
        sorted_labels = [label for label, _ in sorted(Counter(y_true).items(), key=lambda c: (c[0] != O_TOKEN, -c[1]))]
        print("## Confusion matrix for assertions ##")
        print(confusion_matrix(y_true, y_pred, labels=sorted_labels))
        print("\n## Classification report for assertions ##")
        print(classification_report(y_true, y_pred, labels=sorted_labels))
        class_report = classification_report(y_true, y_pred, labels=sorted_labels, output_dict=True)
        return class_report["macro avg"]["f1-score"]

    def evaluate_relation(self) -> float:
        filenames = [filename for filename in os.listdir(self.relation_annotation_dir) if filename.endswith(".rel")]
        relation_results = self._init_dict_results_for_relations()
        for filename in filenames:
            ground_truth_relations = self._load_relation_annotation_file(os.path.join(self.relation_annotation_dir, filename))
            prediction_relations = self._load_relation_annotation_file(os.path.join(self.relation_prediction_dir, filename))

            # Look for true positive
            for gt_rel in ground_truth_relations:
                relation_results[gt_rel.label.value][NB_GROUND_TRUTH] += 1
                for pred_rel in prediction_relations:
                    if self._is_rel_equal(gt_rel, pred_rel) is True:
                        relation_results[gt_rel.label.value][TP] += 1
                        break

            # Count nb of predictions for each relation type
            for pred_rel in prediction_relations:
                relation_results[pred_rel.label.value][NB_PRED] += 1

        for rel_type in relation_results.keys():
            relation_results[rel_type][RECALL] = round(relation_results[rel_type][TP] / relation_results[rel_type][NB_GROUND_TRUTH], 2) if relation_results[rel_type][NB_GROUND_TRUTH] != 0 else 0
            relation_results[rel_type][PRECISION] = round(relation_results[rel_type][TP] / relation_results[rel_type][NB_PRED], 2) if relation_results[rel_type][NB_PRED] != 0 else 0
            recall = relation_results[rel_type][RECALL]
            precision = relation_results[rel_type][PRECISION]
            relation_results[rel_type][F1_SCORE] = round(2*precision*recall/(precision + recall), 2) if precision + recall != 0 else 0

        macro_average = {
            RECALL: round(sum([rel_res[RECALL] for rel_type, rel_res in relation_results.items()])/len(relation_results), 2) if len(relation_results) !=0 else 0,
            PRECISION: round(sum([rel_res[PRECISION] for rel_type, rel_res in relation_results.items()]) / len(relation_results), 2) if len(relation_results) != 0 else 0,
            F1_SCORE: round(sum([rel_res[F1_SCORE] for rel_type, rel_res in relation_results.items()]) / len(relation_results), 2) if len(relation_results) != 0 else 0,
            NB_GROUND_TRUTH: sum([rel_res[NB_GROUND_TRUTH] for rel_type, rel_res in relation_results.items()])
        }
        relation_results["macro_average"] = macro_average
        print("## Classification report for relations ##")
        print("rel_type | precision | recall | f1-score | support\n")
        for rel_type, rel_res in relation_results.items():
            print(f"{rel_type} | {rel_res[PRECISION]} | {rel_res[RECALL]} | {rel_res[F1_SCORE]} | {rel_res[NB_GROUND_TRUTH]}")
        return relation_results["macro_average"][F1_SCORE]


    def _load_entities(self, task_type: str) -> Tuple[List[str], List[str]]:
        ground_truth = []
        predictions = []
        if task_type == TASK_CONCEPT:
            annotation_dir = self.concept_annotation_dir
            prediction_dir = self.concept_prediction_dir
            filenames = [filename for filename in os.listdir(annotation_dir) if filename.endswith(".con")]
        elif task_type == TASK_ASSERTION:
            annotation_dir = self.assertion_annotation_dir
            prediction_dir = self.assertion_prediction_dir
            filenames = [filename for filename in os.listdir(annotation_dir) if filename.endswith(".ast")]
        else:
            raise NotImplementedError(f"Parsing of task {task_type} not implemented")

        for filename in filenames:
            ground_truth_entities = self._load_entity_annotation_file(os.path.join(annotation_dir, filename), task_type)
            prediction_entities = self._load_entity_annotation_file(os.path.join(prediction_dir, filename), task_type)
            entry_tokens = self._load_input_text_file(os.path.join(self.entries_dir, f"{filename.split('.')[0]}.txt"))

            ground_truth.append(self._build_tokens_annotated(entry_tokens, ground_truth_entities))
            predictions.append(self._build_tokens_annotated(entry_tokens, prediction_entities))

        flattened_ground_truth = [
            token_label
            for true in ground_truth for token_label in true
        ]

        flattened_predictions = [
            token_label
            for pred in predictions for token_label in pred
        ]
        return flattened_ground_truth, flattened_predictions

    def _load_relation_annotation_file(self, path: str) -> List[RelationAnnotation]:
        relations_annotated = []
        with open(path, "r") as input_file:
            for rel_line in input_file.readlines():
                rel_annotation = self._parse_relation_annotation(rel_line)
                if rel_annotation is not None:
                    relations_annotated.append(rel_annotation)
                else:
                    print(f"Failed to parse annotation {rel_line} for file {path}")
        return relations_annotated

    def _load_entity_annotation_file(self, path: str, task: str) -> List[EntityAnnotation]:
        entities_annotated = []
        with open(path, "r") as input_file:
            for entity_line in input_file.readlines():
                if task == TASK_CONCEPT:
                    entity_annotation = self._parse_concept_annotation(entity_line)
                elif task == TASK_ASSERTION:
                    entity_annotation = self._parse_assertion_annotation(entity_line)
                else:
                    raise NotImplementedError(f"Parsing of task {task} not implemented")

                if entity_annotation is not None:
                    entities_annotated.append(entity_annotation)
                else:
                    print(f"Failed to parse annotation {entity_line} for file {path}")
        return entities_annotated

    @staticmethod
    def _load_input_text_file(path: str) -> List[List[Token]]:
        tokens = []
        with open(path, "r") as input_file:
            for i, line in enumerate(input_file.readlines()):
                tokens.append([
                    Token(label=O_TOKEN, text=word, line=i, word_index=index) for index, word in enumerate(line.split(" "))
                ])
        return tokens

    @staticmethod
    def _parse_relation_annotation(text: str) -> Optional[RelationAnnotation]:
        try:
            return RelationAnnotation(
                label=RelationValue[text.split("||")[1].split("=")[1].replace('"','').replace("\n","")],
                left_entity=EntityAnnotationForRelation(
                    text=re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[0].split("=")[1].replace('"', ''),
                    start_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[0]),
                    start_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[1]),
                    end_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[0]),
                    end_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[1]),
                ),
                right_entity=EntityAnnotationForRelation(
                    text=re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[2])[0].split("=")[1].replace('"', ''),
                    start_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[2])[1].split(" ")[0].split(":")[0]),
                    start_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[2])[1].split(" ")[0].split(":")[1]),
                    end_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[2])[1].split(" ")[1].split(":")[0]),
                    end_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[2])[1].split(" ")[1].split(":")[1]),
                )
            )
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_concept_annotation(text: str) -> Optional[EntityAnnotation]:
        try:
            return EntityAnnotation(
                label=text.split("||")[1].split("=")[1].replace('"','').replace("\n",""),
                text=re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[0].split("=")[1].replace('"', ''),
                start_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[0]),
                start_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[1]),
                end_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[0]),
                end_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[1])
            )
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_assertion_annotation(text: str) -> Optional[EntityAnnotation]:
        try:
            return EntityAnnotation(
                label=text.split("||")[2].split("=")[1].replace('"', '').replace("\n", ""),
                text=re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[0].split("=")[1].replace('"', ''),
                start_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[0]),
                start_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[0].split(":")[1]),
                end_line=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[0]),
                end_word=int(re.split('(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})', text.split("||")[0])[1].split(" ")[1].split(":")[1])
            )
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _build_tokens_annotated(entry_tokens: List[List[Token]], annotated_entities: List[EntityAnnotation]) -> List[str]:
        copy_entries_tokens = deepcopy(entry_tokens)
        for entity in annotated_entities:
            if entity.start_line == entity.end_line:
                for index in range(entity.start_word, entity.end_word + 1):
                    copy_entries_tokens[entity.start_line - 1][index].label = entity.label
            else:
                for index_line in range(entity.start_line, entity.end_line + 1):
                    if index_line == entity.start_line:
                        for index in range(entity.start_word, len(copy_entries_tokens[index_line - 1]) - 1):
                            copy_entries_tokens[index_line - 1][index].label = entity.label
                    elif index_line == entity.end_line:
                        for index in range(entity.end_word + 1):
                            copy_entries_tokens[index_line - 1][index].label = entity.label
                    else:
                        for index in range(len(copy_entries_tokens[index_line - 1]) - 1):
                            copy_entries_tokens[index_line - 1][index].label = entity.label
        return [
            token.label
            for line in copy_entries_tokens for token in line
        ]

    @staticmethod
    def _init_dict_results_for_relations() -> Dict[str, Any]:
        return {
            relation_type.value : {
                TP: 0,
                NB_GROUND_TRUTH: 0,
                NB_PRED: 0
            }
            for relation_type in RelationValue
        }

    def _is_rel_equal(self, first_rel: RelationAnnotation, snd_rel: RelationAnnotation) -> bool:
        return (first_rel.label == snd_rel.label) and self._is_entity_equal(first_rel.left_entity, snd_rel.left_entity)\
               and self._is_entity_equal(first_rel.right_entity, snd_rel.right_entity)

    @staticmethod
    def _is_entity_equal(first_entity: EntityAnnotationForRelation, snd_entity: EntityAnnotationForRelation) -> bool:
        return (first_entity.start_line == snd_entity.start_line) and (first_entity.end_line == snd_entity.end_line) \
               and (first_entity.start_word == snd_entity.start_word) and \
               (first_entity.end_word == snd_entity.end_word)

if __name__ == "__main__":
    fire.Fire(Evaluator)
