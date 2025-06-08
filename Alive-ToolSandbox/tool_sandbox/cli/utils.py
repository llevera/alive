# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import traceback
from collections import Counter, defaultdict
from enum import auto
from pathlib import Path
from typing import Any, Callable, Optional, Union

from strenum import StrEnum

from tool_sandbox.common.execution_context import (
    RoleType,
    ScenarioCategories,
    get_current_context,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.anthropic_api_agent import (
    ClaudeHaikuAgent,
    ClaudeOpusAgent,
    ClaudeSonnetAgent,
)
from tool_sandbox.roles.azure_openai_agent import (
    AzureGPT4Agent,
    AzureGPT4oAgent,
    AzureGPT4oMiniAgent,
    AzureGPT35TurboAgent,
)
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.cli_role import CliAgent, CliUser
from tool_sandbox.roles.cohere_agent import CohereAgent
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.gemini_agent import GeminiAgent
from tool_sandbox.roles.gorilla_api_agent import GorillaAPIAgent
from tool_sandbox.roles.hermes_api_agent import HermesAPIAgent
from tool_sandbox.roles.mistral_api_agent import MistralOpenAIServerAgent
from tool_sandbox.roles.openai_api_agent import (
    GPT_3_5_0125_Agent,
    GPT_4_0125_Agent,
    GPT_4_o_2024_05_13_Agent,
    GPT_4_o_mini_Agent,
)
from tool_sandbox.roles.openai_api_user import (
    GPT_3_5_0125_User,
    GPT_4_0125_User,
    GPT_4_o_2024_05_13_User,
    GPT_4_o_mini_User,
    AzureGPT4User,
    AzureGPT4oUser,
    AzureGPT4oMiniUser,
    AzureGPT35TurboUser,
)
from tool_sandbox.roles.unhelpful_agent import UnhelpfulAgent
from tool_sandbox.scenarios import named_scenarios


class RoleImplType(StrEnum):
    Hermes = auto()
    Gorilla = auto()
    GPT_3_5_0125 = auto()
    GPT_4_0125 = auto()
    GPT_4_o_2024_05_13 = auto()
    GPT_4_o_mini = auto()
    Azure_GPT_4 = auto()
    Azure_GPT_4o = auto()
    Azure_GPT_4o_mini = auto()
    Azure_GPT_35_Turbo = auto()
    Claude_3_Opus = auto()
    Claude_3_Sonnet = auto()
    Claude_3_Haiku = auto()
    Gemini_1_0 = auto()
    Gemini_1_5 = auto()
    Gemini_1_5_Flash = auto()
    Cli = auto()
    Deterministic = auto()
    MistralOpenAIServer = auto()
    Cohere_Command_R = auto()
    Cohere_Command_R_Plus = auto()
    Unhelpful = auto()


AGENT_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.Hermes: lambda: HermesAPIAgent(
        model_name="NousResearch/Hermes-2-Pro-Mistral-7B"
    ),
    RoleImplType.Gorilla: lambda: GorillaAPIAgent(
        model_name="gorilla-llm/gorilla-openfunctions-v2"
    ),
    RoleImplType.MistralOpenAIServer: lambda: MistralOpenAIServerAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    ),
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_Agent,
    RoleImplType.GPT_4_0125: GPT_4_0125_Agent,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_Agent,
    RoleImplType.GPT_4_o_mini: GPT_4_o_mini_Agent,
    RoleImplType.Azure_GPT_4: AzureGPT4Agent,
    RoleImplType.Azure_GPT_4o: AzureGPT4oAgent,
    RoleImplType.Azure_GPT_4o_mini: AzureGPT4oMiniAgent,
    RoleImplType.Azure_GPT_35_Turbo: AzureGPT35TurboAgent,
    RoleImplType.Claude_3_Opus: ClaudeOpusAgent,
    RoleImplType.Claude_3_Sonnet: ClaudeSonnetAgent,
    RoleImplType.Claude_3_Haiku: ClaudeHaikuAgent,
    RoleImplType.Gemini_1_0: lambda: GeminiAgent(model_name="gemini-1.0-pro"),
    RoleImplType.Gemini_1_5: lambda: GeminiAgent(model_name="gemini-1.5-pro-001"),
    RoleImplType.Gemini_1_5_Flash: lambda: GeminiAgent(
        model_name="gemini-1.5-flash-001"
    ),
    RoleImplType.Cli: CliAgent,
    RoleImplType.Cohere_Command_R: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-v01"
    ),
    RoleImplType.Cohere_Command_R_Plus: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-plus"
    ),
    RoleImplType.Unhelpful: UnhelpfulAgent,
}

USER_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_User,
    RoleImplType.GPT_4_0125: GPT_4_0125_User,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_User,
    RoleImplType.GPT_4_o_mini: GPT_4_o_mini_User,
    RoleImplType.Azure_GPT_4: AzureGPT4User,
    RoleImplType.Azure_GPT_4o: AzureGPT4oUser,
    RoleImplType.Azure_GPT_4o_mini: AzureGPT4oMiniUser,
    RoleImplType.Azure_GPT_35_Turbo: AzureGPT35TurboUser,
    RoleImplType.Cli: CliUser,
}

# The scenarios to play back when the `--test_mode` flag is set.
TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
    # "remove_contact_by_phone_multiple_user_turn",
    # "find_temperature_f_with_location_and_time_diff_multiple_user_turn",
]


def resolve_scenarios_by_category(
    desired_category: str,
    preferred_tool_backend: ToolBackend,
    no_augmentations: bool = False,
) -> dict[str, Scenario]:
    """Resolve scenarios by category.

    Args:
        desired_category: Category of scenarios to run.
        preferred_tool_backend: Which backend should be chosen in face of conflicting tool names.
        no_augmentations: If True, only include scenarios with NO_DISTRACTION_TOOLS category.

    Returns:
        Dictionary from scenario name to definition.
    """
    from tool_sandbox.common.execution_context import ScenarioCategories
    
    # Convert string to ScenarioCategories enum
    try:
        category_enum = ScenarioCategories(desired_category)
    except ValueError:
        available_categories = [cat.value for cat in ScenarioCategories]
        raise ValueError(
            f"Invalid category '{desired_category}'. Available categories: {available_categories}"
        )
    
    all_scenarios = named_scenarios(preferred_tool_backend=preferred_tool_backend)
    
    name_to_scenario = {}
    for name, scenario in all_scenarios.items():
        # Check if scenario has the desired category
        if category_enum in scenario.categories:
            # If no_augmentations is True, only include scenarios with NO_DISTRACTION_TOOLS
            if no_augmentations:
                if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
                    name_to_scenario[name] = scenario
            else:
                name_to_scenario[name] = scenario
    
    return name_to_scenario


def resolve_scenarios(
    desired_scenario_names: Optional[list[str]],
    preferred_tool_backend: ToolBackend,
    desired_category: Optional[str] = None,
    no_augmentations: bool = False,
) -> dict[str, Scenario]:
    """Resolve the scenarios to run.

    Args:
        desired_scenario_names: Name of scenarios to run. If empty all scenarios will be
                                returned.
        preferred_tool_backend: Which backend should be chosen in face of conflicting tool names.
        desired_category: Category of scenarios to run. If provided, filters by category.
        no_augmentations: If True, only include scenarios with NO_DISTRACTION_TOOLS category.

    Returns:
        Dictionary from scenario name to definition.
    """
    # If category is specified, use category-based filtering
    if desired_category is not None:
        return resolve_scenarios_by_category(
            desired_category=desired_category,
            preferred_tool_backend=preferred_tool_backend,
            no_augmentations=no_augmentations,
        )
    
    if desired_scenario_names is None:
        # No filtering needed. Return all scenarios.
        all_scenarios = named_scenarios(preferred_tool_backend=preferred_tool_backend)
        if no_augmentations:
            # Filter to only include scenarios with NO_DISTRACTION_TOOLS
            from tool_sandbox.common.execution_context import ScenarioCategories
            return {
                name: scenario
                for name, scenario in all_scenarios.items()
                if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories
            }
        return all_scenarios

    name_to_scenario = {
        name: scenario
        for name, scenario in named_scenarios(
            preferred_tool_backend=preferred_tool_backend
        ).items()
        if name in desired_scenario_names
    }

    # Raise an exception if not all desired scenarios exist, e.g. to fail if there was a
    # typo in the scenario names of the CLI command.
    if len(desired_scenario_names) != len(name_to_scenario):
        missing_scenarios = set(desired_scenario_names) - set(name_to_scenario.keys())
        raise KeyError(
            "The following desired scenarios do not exist: "
            f"{sorted(list(missing_scenarios))}"
        )
    return name_to_scenario


def run_scenario(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
) -> dict[str, Any]:
    """Play and evaluate a scenario.

    This is a necessary utility function to make multiprocessing work.

    Args:
        name_and_scenario:              Scenario name and Scenario object.
        agent_type:                     Agent type.
        user_type:                      User type.
        output_directory:               Directory to write output into.

    Returns:
        Evaluation info
    """
    name, scenario = name_and_scenario
    roles = {
        RoleType.USER: USER_TYPE_TO_FACTORY[user_type](),
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        RoleType.AGENT: AGENT_TYPE_TO_FACTORY[agent_type](),
    }
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        result = scenario.play_and_evaluate(
            roles=roles,
            output_directory=output_directory,
            scenario_name=name,
        )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": None,
            "exception_type": None,
            "milestone_similarity": result.evaluation_result.milestone_similarity,
            "minefield_similarity": result.evaluation_result.minefield_similarity,
            "similarity": result.evaluation_result.similarity,
            "turn_count": result.evaluation_result.turn_count,
            "milestone_mapping": result.evaluation_result.milestone_mapping,
            "minefield_mapping": result.evaluation_result.minefield_mapping,
        }
    except Exception as e:
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": traceback.format_exc(),
            "exception_type": type(e).__name__,
            "milestone_similarity": 0,
            "minefield_similarity": 0,
            "similarity": 0,
            "turn_count": scenario.max_messages,
            "milestone_mapping": {},
            "minefield_mapping": {},
        }
    finally:
        for role in roles.values():
            role.teardown()


def get_category_summary(
    result_summary: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per test case result summary into category wise summary.

    Args:
        result_summary:     A list of results for each test case.

    Returns:
        Category wise summary.
    """
    # Aggregate results by category
    category_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for current_summary in result_summary:
        for category in current_summary["categories"]:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                current_summary["categories"]
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_summary[category]["similarity"].append(
                current_summary["similarity"]
            )
            category_summary[category]["turn_count"].append(
                current_summary["turn_count"]
            )
        category_summary["ALL_CATEGORIES"]["similarity"].append(
            current_summary["similarity"]
        )
        category_summary["ALL_CATEGORIES"]["turn_count"].append(
            current_summary["turn_count"]
        )
    return category_summary


def get_category_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on ScenarioCategories.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each category.
    """
    category_counter: Counter[Union[ScenarioCategories, str]] = Counter()
    for scenario in name_to_scenario.values():
        for category in scenario.categories:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                scenario.categories
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_counter[category] += 1
        category_counter["ALL_CATEGORIES"] += 1
    return category_counter


def get_necessary_tool_name_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on necessary tool names.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each necessary tool names.
    """
    tool_name_counter: Counter[Union[ScenarioCategories, str]] = Counter(
        {
            tool_name: 0
            for tool_name in get_current_context().get_available_tools(
                scrambling_allowed=False
            )
        }
    )
    # Necessary tool names can be deducted from allowed tools in NO_DISTRACTION_TOOLS category
    # Then the total count equals the count from this category * number of augmentations.
    augmentation_categories: set[Union[ScenarioCategories, str]] = set()
    for scenario in name_to_scenario.values():
        if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
            assert scenario.starting_context.tool_allow_list is not None
            for necessary_tool in scenario.starting_context.tool_allow_list:
                tool_name_counter[necessary_tool] += 1
        augmentation_categories |= {
            ScenarioCategories.NO_DISTRACTION_TOOLS,
            ScenarioCategories.THREE_DISTRACTION_TOOLS,
            ScenarioCategories.TEN_DISTRACTION_TOOLS,
            ScenarioCategories.ALL_TOOLS_AVAILABLE,
            ScenarioCategories.TOOL_NAME_SCRAMBLED,
            ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_TYPE_SCRAMBLED,
            ScenarioCategories.ARG_NAME_SCRAMBLED,
        } & set(scenario.categories)
    for necessary_tool in tool_name_counter:
        tool_name_counter[necessary_tool] *= len(augmentation_categories)
    return tool_name_counter
