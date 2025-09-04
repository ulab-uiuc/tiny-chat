from typing import cast

from litellm import acompletion, completion
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.utils import supports_response_schema
from pydantic import validate_call
from rich import print

from tiny_chat.messages import (
    ActionType,
    AgentAction,
    ScriptBackground,
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from tiny_chat.profiles import BaseEnvironmentProfile, BaseRelationshipProfile
from tiny_chat.utils import format_docstring
from tiny_chat.utils.logger import logger as log
from tiny_chat.utils.template import (
    ACTION_NORMAL_TEMPLATE,
    ACTION_SCRIPT_TEMPLATE,
    BAD_OUTPUT_REFORMAT_TEMPLATE,
    ENV_PROFILE_TEMPLATE,
    FIRST_PERSON_NARRATIVE_TEMPLATE,
    GOAL_TEMPLATE,
    INIT_PROFILE_TEMPLATE,
    RELATIONSHIP_PROFILE_TEMPLATE,
    SCRIPT_FULL_TEMPLATE,
    SCRIPT_SINGLE_STEP_TEMPLATE,
    SECOND_PERSON_NARRATIVE_TEMPLATE,
)

from .output_parsers import (
    EnvResponse,
    OutputParser,
    OutputType,
    PydanticOutputParser,
    ScriptOutputParser,
    StrOutputParser,
)


def _prepare_provider_config(model_name: str) -> tuple[str | None, str | None, str]:
    """兼容性函数 - 使用providers.utils中的统一实现"""
    from tiny_chat.providers.utils import prepare_model_config_from_name

    return prepare_model_config_from_name(model_name)


DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"


@validate_call
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
) -> str:
    template = BAD_OUTPUT_REFORMAT_TEMPLATE

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)
    response = await acompletion(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    log.info(f"Reformatted output: {reformatted_output}")
    return reformatted_output


@validate_call
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: float = 0.7,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    max_parse_retries: int = 3,
) -> OutputType:
    """Generate text using LiteLLM instead of Langchain."""
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    template = format_docstring(template)

    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    api_base, api_key, model_name = _prepare_provider_config(model_name)

    if structured_output:
        if not api_base:
            params = get_supported_openai_params(model=model_name)
            assert params is not None
            assert (
                "response_format" in params
            ), "response_format is not supported in this model"
            assert supports_response_schema(
                model=model_name
            ), "response_schema is not supported in this model"
        messages = [{"role": "user", "content": template}]

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"
        response = await acompletion(
            model=model_name,
            messages=messages,
            response_format=output_parser.pydantic_object,
            drop_params=True,
            temperature=temperature,
            base_url=api_base,
            api_key=api_key,
        )
        result = response.choices[0].message.content
        log.info(f"Generated result: {result}")
        assert isinstance(result, str)
        return cast(OutputType, output_parser.parse(result))

    messages = [{"role": "user", "content": template}]

    response = await acompletion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        drop_params=True,
        base_url=api_base,
        api_key=api_key,
    )
    result = response.choices[0].message.content

    max_retries = max_parse_retries
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                parsed_result = output_parser.parse(result)
            else:
                log.debug(
                    f"[yellow] Parse attempt {attempt + 1}/{max_retries}: Reformatting result",
                    extra={"markup": True},
                )
                reformat_result = await format_bad_output(
                    result,
                    output_parser.get_format_instructions(),
                    bad_output_process_model or model_name,
                    use_fixed_model_version,
                )
                parsed_result = output_parser.parse(reformat_result)

            log.info(f"Generated result (attempt {attempt + 1}): {parsed_result}")
            return parsed_result

        except Exception as e:
            if isinstance(output_parser, ScriptOutputParser):
                raise e

            if attempt == max_retries - 1:
                log.error(
                    f"[red] Failed to parse result after {max_retries} attempts. "
                    f"Last error: {e}\nFinal result: {result}",
                    extra={"markup": True},
                )
                raise e
            else:
                log.debug(
                    f"[red] Parse attempt {attempt + 1} failed: {e}\nRetrying...",
                    extra={"markup": True},
                )
                continue

    raise RuntimeError(f"Unexpected error in parsing after {max_retries} attempts")


@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float = 0.7,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> BaseEnvironmentProfile:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template=ENV_PROFILE_TEMPLATE,
        input_values={
            "inspiration_prompt": inspiration_prompt,
            "examples": examples,
        },
        output_parser=PydanticOutputParser(pydantic_object=BaseEnvironmentProfile),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[BaseRelationshipProfile, str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    return await agenerate(
        model_name=model_name,
        template=RELATIONSHIP_PROFILE_TEMPLATE,
        input_values={
            "agent_profile": agent_profile,
        },
        output_parser=PydanticOutputParser(pydantic_object=BaseRelationshipProfile),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
def generate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: float = 0.7,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    max_parse_retries: int = 3,
) -> AgentAction:
    """
    Synchronous version of agenerate_action
    """
    try:
        if script_like:
            # model as playwright
            template = ACTION_SCRIPT_TEMPLATE
        else:
            template = ACTION_NORMAL_TEMPLATE

        api_base, api_key, model_name = _prepare_provider_config(model_name)
        output_parser: PydanticOutputParser[AgentAction] = PydanticOutputParser(
            pydantic_object=AgentAction
        )
        format_instructions = output_parser.get_format_instructions()
        messages = [
            {
                "role": "user",
                "content": template.format(
                    agent=agent,
                    turn_number=str(turn_number),
                    history=history,
                    action_list=" ".join(action_types),
                    format_instructions=format_instructions,
                ),
            }
        ]

        response = completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            drop_params=True,
            base_url=api_base,
            api_key=api_key,
        )
        result = response.choices[0].message.content
        assert isinstance(result, str)
        return output_parser.parse(result)
    except Exception as e:
        log.warning(f"Failed to generate action due to {e}")
        return AgentAction(action_type="none", argument="")


@validate_call
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: float = 0.7,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    max_parse_retries: int = 3,
) -> AgentAction:
    """
    Using langchain to generate an example episode
    """
    try:
        if script_like:
            # model as playwright
            template = ACTION_SCRIPT_TEMPLATE
        else:
            template = ACTION_NORMAL_TEMPLATE
        return await agenerate(
            model_name=model_name,
            template=template,
            input_values={
                "agent": agent,
                "turn_number": str(turn_number),
                "history": history,
                "action_list": " ".join(action_types),
            },
            output_parser=PydanticOutputParser(pydantic_object=AgentAction),
            temperature=temperature,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            max_parse_retries=max_parse_retries,
        )
    except Exception as e:
        log.warning(f"Failed to generate action due to {e}")

        async def return_default() -> AgentAction:
            return AgentAction(action_type="none", argument="")

        return await return_default()


@validate_call
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: float = 0.7,
    agent_names: list[str] | None = None,
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            return await agenerate(
                model_name=model_name,
                template=SCRIPT_SINGLE_STEP_TEMPLATE,
                input_values={
                    "background": background.to_natural_language(),
                    "history": history,
                    "agent": agent_name,
                },
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names or [],
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )

        else:
            return await agenerate(
                model_name=model_name,
                template=SCRIPT_FULL_TEMPLATE,
                input_values={
                    "background": background.to_natural_language(),
                },
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names or [],
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@validate_call
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@validate_call
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template=INIT_PROFILE_TEMPLATE,
        input_values={
            "name": basic_info["name"],
            "age": basic_info["age"],
            "gender_identity": basic_info["gender_identity"],
            "pronoun": basic_info["pronoun"],
            "occupation": basic_info["occupation"],
            "bigfive": basic_info["Big_Five_Personality"],
            "mft": basic_info["Moral_Foundation"],
            "schwartz": basic_info["Schwartz_Portrait_Value"],
            "decision_style": basic_info["Decision_making_Style"],
            "secret": basic_info["secret"],
        },
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template=FIRST_PERSON_NARRATIVE_TEMPLATE,
            input_values={"text": text},
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template=SECOND_PERSON_NARRATIVE_TEMPLATE,
            input_values={"text": text},
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@validate_call
def generate_goal(
    model_name: str,
    background: str,
    temperature: float = 0.7,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Synchronous version of agenerate_goal
    """
    api_base, api_key, model_name = _prepare_provider_config(model_name)
    messages = [
        {
            "role": "user",
            "content": GOAL_TEMPLATE.format(background=background),
        }
    ]

    response = completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        drop_params=True,
        base_url=api_base,
        api_key=api_key,
    )
    result = response.choices[0].message.content
    assert isinstance(result, str)
    return result


@validate_call
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    return await agenerate(
        model_name=model_name,
        template=GOAL_TEMPLATE,
        input_values={"background": background},
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )
