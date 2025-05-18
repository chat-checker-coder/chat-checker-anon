import json
from pathlib import Path
import os
import yaml
from typing import List, Optional

from rich import print
from litellm import completion
from litellm.types.utils import ModelResponse, Choices


from chat_checker.data_management.storage_manager import load_user_personas
from chat_checker.models.chatbot import Chatbot
from chat_checker.models.user_personas import GeneratedPersonas, Persona, PersonaType
from chat_checker.utils.llm_utils import supports_structured_outputs, DEFAULT_LLM
from chat_checker.utils.misc_utils import get_matching_api_key
from chat_checker.persona_generation.persona_gen_prompts import (
    standard_persona_description,
    challenging_persona_description,
    adversarial_persona_description,
    persona_generation_prompt,
)

# Build the path to the .env file
CHAT_CHECKER_BASE_DIR = Path(__file__).parent


def gen_personas(
    chatbot: Chatbot,
    num_personas: int = 1,
    persona_type: PersonaType = PersonaType.STANDARD,
    start_num=1,
    model: str = DEFAULT_LLM,
    seed: Optional[int] = None,
    save_prompt: bool = True,
) -> List[Persona]:
    assert supports_structured_outputs(model)

    persona_type_description = ""
    if persona_type == PersonaType.STANDARD:
        persona_type_description = standard_persona_description
    elif persona_type == PersonaType.CHALLENGING:
        persona_type_description = challenging_persona_description
    elif persona_type == PersonaType.ADVERSARIAL:
        persona_type_description = adversarial_persona_description

    prompt = persona_generation_prompt.format(
        num_personas=num_personas,
        persona_type=persona_type,
        chatbot_info=chatbot.info.dump_as_yaml_without_task(),
        persona_type_description=persona_type_description,
    )

    # Write prompt to a file for debugging
    if save_prompt:
        persona_prompts_dir = (
            f"{chatbot.base_directory}/user_personas/persona_gen_prompts"
        )
        os.makedirs(persona_prompts_dir, exist_ok=True)
        with open(
            f"{persona_prompts_dir}/gen_{num_personas}_{persona_type}_personas.txt",
            "w+",
            encoding="utf-8",
        ) as f:
            f.write(prompt)

    response: ModelResponse = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=GeneratedPersonas,
        seed=seed,
        api_key=get_matching_api_key(model).get_secret_value(),
        drop_params=True,
    )
    # for type-checking
    assert isinstance(response, ModelResponse)
    assert isinstance(response.choices[0], Choices)
    if not response.choices[0].message.content:
        raise ValueError("Missing generated personas")

    personas_json = json.loads(response.choices[0].message.content)
    generated_personas = GeneratedPersonas(**personas_json)

    if not generated_personas:
        raise ValueError("Failed to generate personas.")

    personas = []
    for i, persona in enumerate(generated_personas.personas):
        persona_num = f"{i + start_num:02d}"
        persona_model = Persona(
            persona_id=f"generated_{persona_type}_persona_{persona_num}",
            type=persona_type,
            profile={
                "name": persona.name,
                "gender": persona.gender,
                "age": persona.age,
                "background_info": persona.background_info,
                "personality": persona.personality,
                "interaction_style": persona.interaction_style,
            },
            task=persona.task,
        )
        personas.append(persona_model)
    return personas


def run(
    chatbot: Chatbot,
    persona_type: PersonaType = PersonaType.STANDARD,
    num_personas: int = 1,
    verbose: bool = False,
    seed: Optional[int] = None,
    save_prompt: bool = True,
):
    existing_personas = load_user_personas(chatbot)
    existing_generated_personas_of_type = [
        persona
        for persona in existing_personas.values()
        if persona.type == persona_type and persona.generated
    ]
    persona_dir = chatbot.base_directory / "user_personas"

    # Find out start number for generated personas
    os.makedirs(persona_dir, exist_ok=True)
    start_num = len(existing_generated_personas_of_type) + 1
    if verbose:
        print(f"Start number for generated personas: {start_num}")

    print(
        f"Generating {num_personas} {persona_type} personas for chatbot {chatbot.id}..."
    )

    model = os.getenv("CHAT_CHECKER_PERSONA_GEN_LLM", DEFAULT_LLM)
    personas = gen_personas(
        chatbot=chatbot,
        num_personas=num_personas,
        persona_type=persona_type,
        start_num=start_num,
        model=model,
        seed=seed,
        save_prompt=save_prompt,
    )
    print(f"Generated {len(personas)} personas")
    if verbose:
        print(personas)

    print(f"Saving personas to {persona_dir}...")
    for persona in personas:
        persona_id = persona.persona_id
        persona_file = persona_dir / f"{persona_id}.yaml"
        with open(persona_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                persona.model_dump(), f, indent=4, allow_unicode=True, sort_keys=False
            )
    print("Personas saved successfully")
