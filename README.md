# AI Story Data Generation

An intelligent story data generation system based on large language models (LLMs), supporting multilingual (Chinese/English) story creation, plot generation, quality evaluation, and user interaction simulation.

## Project Overview

This project is a complete AI-driven story generation pipeline that can:
- Automatically generate high-quality story content
- Support multi-episode storyline management
- Provide story quality evaluation and optimization
- Simulate user interaction behaviors
- Support bilingual generation in Chinese and English

## Core Architecture

### Main Entry File
- **`pipeline_auto.py`** – The main entry of the project, containing the full story generation pipeline  
  - `StoryOrganizer`: Story structure management  
  - `HistoryManager`: Dialogue history management  
  - `StoryGenerator`: Core story generator  

### Core Functional Modules

#### 1. Story Generation Module
- **`generate.py`** – Main story content generator  
  - Generates story plots and dialogues based on LLMs  
  - Supports trigger condition management  
  - Multi-turn dialogue generation  

- **`user_generate.py`** – User behavior simulator  
  - Generates user inputs and interaction behaviors  
  - Supports normal and abnormal user behavior patterns  
  - Intelligent role-playing  

#### 2. Quality Optimization Module
- **`refine.py`** – Story quality evaluator and optimizer  
  - Multi-dimensional quality scoring  
  - Content optimization suggestions  
  - Automatic quality improvement  

- **`regenerate.py`** – Story regeneration module  
  - Regenerates content based on feedback  
  - Maintains story coherence  
  - Quality enhancement mechanisms  

#### 3. Story Management Module
- **`story_summary.py`** – Story summary generator  
  - Automatically generates episode summaries  
  - Organizes story progression  
  - Extracts key information  

- **`story_trigger.py`** – Story trigger condition manager  
  - Episode transition condition management  
  - Plot triggering logic  
  - Branching storyline control  

#### 4. Infrastructure Module
- **`llm.py`** – Large language model interface layer  
  - Supports multiple LLM services (OpenAI, Azure, DeepSeek, GPT Proxy)  
  - Unified API interface  
  - Streaming output support  
  - Error retry mechanisms  

- **`template_factory.py`** – Template factory  
  - Chinese and English template management  
  - Dynamic template generation  
  - Support for multiple template types  

- **`utils.py`** – Utility function library  
  - JSON parsing utilities  
  - Text processing functions  
  - Colored output support  

- **`my_threading.py`** – Multithreading support  
  - Concurrent processing capability  
  - Context management  

## Storyline Generator Module (`storyline_generator/`)

### Core Files
- **`competent_create_storyline.py`** – Intelligent storyline creator  
  - Generates story outlines based on keywords  
  - Multi-episode plot design  
  - Story structure optimization  

- **`competent_extra_story_info.py`** – Story information expander  
  - Character profile generation  
  - World-building  
  - Background story enrichment  

- **`pipeline_gen_game_plot.py`** – Game plot generation pipeline  
  - Game-oriented story design  
  - Interactive plot generation  
  - Branching choice management  

### Data Directories
- **`story_framework_data/`** – Story framework data  
- **`source_stories/`** – Source story data  

### Development Tools
- **`test.ipynb`** – Testing and debugging notebook  
- **`gen_game_plot.ipynb`** – Game plot generation experiments  
- **`gen_novel_plot.ipynb`** – Novel plot generation experiments  
- **`plot_refine.ipynb`** – Plot refinement experiments  

### Template Files
- **`templates_zh.py`** – Chinese template library (102 KB, 1133 lines)  
- **`templates_en.py`** – English template library (96 KB, 1400 lines)  

## Features

### Multilingual Support
- Full Chinese and English bilingual support  
- Localized templates and prompts  
- Language-specific text processing  

### Intelligent Story Generation
- Creative content generation powered by LLMs  
- Multi-turn dialogue and plot progression  
- Character consistency and narrative coherence  

### Quality Assurance
- Multi-dimensional quality evaluation  
- Automatic content optimization  
- User feedback integration  

### Flexible Configuration
- Supports multiple LLM providers  
- Configurable generation parameters  
- Modular design for easy extension  

### Concurrent Processing
- Multithreading support  
- Batch story generation  
- Efficient resource utilization  

## Usage

### Basic Usage

```python
from pipeline_auto import generate_story

generate_story(
    story_file="path/to/story_config.json",
    story_output_dir="output/stories",
    refine_output_file="output/refined_stories.jsonl",
    language="en"  # or "zh"
)
```

### Parallel Generation

```python
from pipeline_auto import generate_story_parallel

generate_story_parallel(
    story_file="path/to/story_config.json",
    story_output_dir="output/stories",
    refine_output_dir="output/refined",
    num_threads=4,
    num_runs=10
)
```

### Story Configuration Format

```json
{
  "Settings": {
    "story_id": "unique_story_id",
    "story_name": "Story Title",
    "story_style": ["Adventure", "Mystery"],
    "story_desc": "Story description",
    "story_chars": {
      "Protagonist": "Character description"
    },
    "leading_role": "Protagonist",
    "story_goal": "Overall story goal",
    "states": [["1", "2", "3"], ["1", "4", "5"]]
  },
  "Episodes": [
    {
      "episode_id": "1",
      "episode_desc": "Episode description",
      "episode_goal": "Episode goal",
      "episode_scene": "Scene description",
      "episode_chars": ["Character1", "Character2"],
      "pre_episode_id": [],
      "triggers": [
        {
          "condition": "Trigger condition",
          "next_episode": "2"
        }
      ]
    }
  ]
}
```

## Environment Requirements

### Python Dependencies

```
colorama
jsonlines
loguru
more_itertools
requests
tqdm
langchain
```

### LLM Service Configuration
The project supports multiple LLM services and requires API keys to be configured in the corresponding modules:

- OpenAI API  
- Azure OpenAI  
- DeepSeek API  
- GPT Proxy  

## Project Structure

```
ai-story-data-generation/
├── pipeline_auto.py
├── generate.py
├── user_generate.py
├── refine.py
├── regenerate.py
├── story_summary.py
├── story_trigger.py
├── llm.py
├── template_factory.py
├── utils.py
├── my_threading.py
├── templates_zh.py
├── templates_en.py
├── data/
├── storyline_generator/
│   ├── competent_create_storyline.py
│   ├── competent_extra_story_info.py
│   ├── pipeline_gen_game_plot.py
│   ├── story_framework_data/
│   ├── source_stories/
│   └── *.ipynb
└── README.md
```
## Reinforcement Learning Module (`RL/`)

The **`EasyR1-v2/`** directory contains the implementation of **Controllable Reinforcement Learning (Controllable RL)** for story generation and refinement.  
This module focuses on optimizing long-horizon narrative structure and controllable story attributes by introducing reinforcement learning signals aligned with explicit control objectives.


## Development Guide

### Adding a New LLM Service
1. Inherit from the `BaseLLM` class in `llm.py`  
2. Implement the `chat` and `set_para` methods  
3. Configure usage in the relevant modules  

### Extending the Template System
1. Add new templates in `templates_zh.py` or `templates_en.py`  
2. Register templates in `template_factory.py`  
3. Update the corresponding generator modules  

### Customizing Quality Evaluation
1. Modify the evaluation logic in `refine.py`  
2. Adjust scoring weights and criteria  
3. Integrate new evaluation dimensions  

## Contribution Guidelines

1. Fork the project  
2. Create a feature branch  
3. Commit your changes  
4. Submit a Pull Request  

## License

[To be added]

## Contact

[To be added]

---

**Note**: Before use, please ensure that API keys for the selected LLM services are properly configured and generation parameters are adjusted according to your needs.
