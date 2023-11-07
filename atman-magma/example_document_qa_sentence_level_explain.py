from atman_magma.magma import Magma
from atman_magma.document_qa_explainer import DocumentQAExplainer

DOCUMENT = '''Name: Donald Robertson
Email: donald.robertson@gmail.com
Phone: 609-440-9978

Objective
Talented and communicative game designer with 6+ years of experience in a creative yet deadline-driven environment. Eager to join The Rock Studios to help manage the design of gameplay systems and online mechanics. In previous roles designed and co-wrote 5 AAA titles and created more than 200 levels within the action-RPG game area.

Work Experience
Game Designer
NextGen Games, Los Angeles, CA
2016-2018

Designed and drove the vision and implementation of features and game systems.
Developed ideas for gameplay throughout the player life-cycle.
Worked within existing systems and expanded them, including integrating feedback from UX testing.
Partnered with other teams to consider the game design and player engagement targets.
Created and tuned content to create accessible but deep experiences for players.
Key achievements:

Designed and co-wrote 5 AAA released game titles.
Improved player success rate by 150% as a result of close collaboration with the UX team and developing a set of measurable tests and questionnaires.

Gameplay Designer
Q2 BFG, Los Angeles, CA
2013–2016

Built gameplay scenarios in various styles and for different purposes.
Created gameplay moments, including narrative events, combat encounters, and points of interest within the game world.
Collaborated with partner teams to ensure a cohesive and coherent scenario experience.
Worked with engineers on developing and maintaining scenario building functionality that redefined design boundaries.
Authored and reviewed design documentation.
Key achievement:
Developed a data-gathering method for game balancing and tuning.
Created over 200 levels for various Sci-Fi and Fantasy Action-RPG games.

Education
MS, Computer Science/Game Design
University of Southern California, School of Cinematic Arts
2013'''

print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

document_explainer = DocumentQAExplainer(
    model,
    explanation_delimiters = ['\n'],  ## for now max len = 1 token (todo: add support for longer delimiters)
    device = 'cuda:0',
    suppression_factor = 0.0,  ## suppression_factor = 0.0 -> completely erases one chunk at a time
    document = DOCUMENT
)

question = 'What is the key achievement of the candidate?'
expected_answer = ' built a balancing utility'

output = document_explainer.run(
    question = question,
    expected_answer = expected_answer,
    max_batch_size = 25
)

## plt imshow output
document_explainer.show_output(
    output = output,
    figsize=(14,19),
    fontsize = 20,
    question = question,
    expected_answer = expected_answer,
)

## save as json or print
postprocessed_output = document_explainer.postprocess(output)
postprocessed_output.save_as(filename = 'explanation.json')
print('explanation.json')
