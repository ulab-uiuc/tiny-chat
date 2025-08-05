# """
# Prompt templates for tiny-chat generation tasks.

# This module contains all the prompt templates used in the tiny-chat project
# for various generation tasks such as environment profiles, relationship profiles,
# actions, scripts, and more.
# """

# Bad output reformatting prompt
BAD_OUTPUT_REFORMAT_TEMPLATE = """
Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
Original string: {ill_formed_output}

Format instructions: {format_instructions}

Please only generate the JSON:
"""

# Environment profile generation prompt
ENV_PROFILE_TEMPLATE = """Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
Examples:
{examples}
Inspirational prompt: {inspiration_prompt}
Please use the following format:
{format_instructions}
"""

# Relationship profile generation prompt
RELATIONSHIP_PROFILE_TEMPLATE = """Please generate relationship between two agents based on the agents' profiles below. Note that you generate
{agent_profile}
Please use the following format:
{format_instructions}
"""

# Action generation prompt (script-like mode)
ACTION_SCRIPT_TEMPLATE = """
Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
{history}.
The script has proceeded to Turn #{turn_number}. Current available action types are
{action_list}.
Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{format_instructions}
"""

# Action generation prompt (normal mode)
ACTION_NORMAL_TEMPLATE = """
Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent}'s goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
{history}.
You are at Turn #{turn_number}. Your available action types are
{action_list}.
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{format_instructions}
"""

# Script generation prompt (single step)
SCRIPT_SINGLE_STEP_TEMPLATE = """Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

Here are the conversation background and history:
{background}
{history}

Remember that you are an independent scriptwriter and should finish the script by yourself.
The output should only contain the script following the format instructions, with no additional comments or text.

Here are the format instructions:
{format_instructions}"""

# Script generation prompt (full script)
SCRIPT_FULL_TEMPLATE = """
Please write the script between two characters based on their social goals with a maximum of 20 turns.

{background}
Your action should follow the given format:
{format_instructions}
Remember that you are an independent scriptwriter and should finish the script by yourself.
The output should only contain the script following the format instructions, with no additional comments or text."""

# Initial profile generation prompt
INIT_PROFILE_TEMPLATE = """Please expand a fictional background for {name}. Here is the basic information:
    {name}'s age: {age}
    {name}'s gender identity: {gender_identity}
    {name}'s pronouns: {pronoun}
    {name}'s occupation: {occupation}
    {name}'s big 5 personality traits: {bigfive}
    {name}'s moral Foundation: think {mft} is more important than others
    {name}'s Schwartz portrait value: {schwartz}
    {name}'s decision-making style: {decision_style}
    {name}'s secret: {secret}
    Include the previous information in the background.
    Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
    For the personality and values (e.g., MBTI, moral foundation, and etc.),
    remember to use examples and behaviors in the person's life to demonstrate it.
    """

# First person narrative conversion prompt
FIRST_PERSON_NARRATIVE_TEMPLATE = """Please convert the following text into a first-person narrative.
e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
{text}"""

# Second person narrative conversion prompt
SECOND_PERSON_NARRATIVE_TEMPLATE = """Please convert the following text into a second-person narrative.
e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
{text}"""

# Goal generation prompt
GOAL_TEMPLATE = """Please generate your goal based on the background:
    {background}
    """ 
