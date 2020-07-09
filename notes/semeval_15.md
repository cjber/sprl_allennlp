# SemEval-2015 Task 8: SpaceEval

## INTRODUCTION

Natural languages are filled with particular constructions for talking about spatial information, including toponyms, spatial nominals, locations that are described in relation to other locations, and movements along a path. The goals of this task include identifying and classifying items from an inventory of spatial concepts:

* Locations: regions, spatial objects, geographic and geopolitical places.
* Entities participating in spatial relations.
* Paths: routes, lines, turns, arcs.
* Topological relations: in, connected.
* Direction and Orientation: North, down.
* Time and space measurements: 20 miles away, for two hours.
* Object properties: intrinsic orientation, dimensionality.
* Frames of Reference: absolute, intrinsic, relative.
* Motion: tracking objects over time.
 
## TASK DESCRIPTION

SpaceEval will build on the spatial role identification tasks as introduced in SemEval 2012 and used in SemEval 2013. SpacEval will consist of the following subtasks, adopting the annotation specification from ISO-Space:

1. Spatial Elements (SE):
    - Identify spans of spatial elements including locations, paths, events and other spatial entities.
    - Classify spatial elements according to type: PATH, PLACE, MOTION, NONMOTION_EVENT, SPATIAL_ENTITY.
    - Identify their attributes according to type.

2. Spatial Signal Identification (SS):
    - Identify spans of spatial signals.
    - Identify their attributes.

3.  Motion Signal Identification (MI):
    - Identify spans of path-of-motion and manner-of-motion signals).
    - Identify their attributes.

4. Motion Relation Identification (MoveLink):
    - Identify relations between motion-event triggers, motion signals, and motion-event participants).
    - Identify their attributes.
5. Spatial Configuration Identification (QSLink):
    - Identify qualitative spatial relations between spatial signals and spatial elements.
    - Identify their attributes.
6. Spatial Orientation Identification (OLink):
    - Identify orientational relations between spatial signals and spatial elements.
    - Identify their attributes.
 
## EVALUATION

There will be three separate evaluation configurations for participants:

Only unannotated text is given to the user.
Manually annotated spatial element extents (no attributes) are given.
Manually annotated spatial element extents and their attributes are given.
Evaluation for each configuration is defined as follows:

1. Only Unannotated Text:
    - SE.a: precision, recall, and F1.
    - SE.b: precision, recall, and F1 for each type, and an overall precision, recall, and F1.
    - SE.c: precision, recall, and F1 for each attribute, and an overall precision, recall, and F1.
    - MoveLink.a, QSLink.a, OLink.a: precision, recall, and F1.
    - MoveLink.b, QSLink.b, OLink.b: precision, recall, and F1 for each attribute, and an overall precision, recall, and F1.

2. Spatial Elements are Provided:
    - SE.b and SE.c: precision, recall, and F1 for each type and its attributes, and an overall precision, recall, and F1.
    - MoveLink.a, QSLink.a, OLink.a: precision, recall, and F1.
    - MoveLink.b, QSLink.b, OLink.b: precision, recall, and F1 for each attribute, and an overall precision, recall, and F1.

3. Spatial Elements, their Types, and their Attributes are Provided:
    - MoveLink.a, QSLink.a, OLink.a: precision, recall, and F1.
    - MoveLink.b, QSLink.b, OLink.b: precision, recall, and F1 for each attribute, and an overall precision, recall, and F1.
