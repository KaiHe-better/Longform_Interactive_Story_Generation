import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """官方的last token pooling方法"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def compare_embeddings_detailed(text, model1_name, model2_name):
    """
    详细对比两个Qwen3模型的embedding和hidden states
    
    Args:
        text: 输入文本
        model1_name: Qwen3-0.6B模型
        model2_name: Qwen3-Embedding-0.6B模型
    """
    print("=" * 80)
    print("Qwen3 模型详细对比")
    print("=" * 80)
    print(f"模型1: {model1_name}")
    print(f"模型2: {model2_name}")
    print(f"输入文本: {text}")
    print("-" * 80)
    
    # 加载模型和tokenizer
    # 注意：Embedding模型使用left padding
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name, trust_remote_code=True)
    model1 = AutoModel.from_pretrained(model1_name, trust_remote_code=True, output_hidden_states=True)
    
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name, trust_remote_code=True, padding_side='left')
    model2 = AutoModel.from_pretrained(model2_name, trust_remote_code=True, output_hidden_states=True)
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        # 模型1 (普通Qwen3)
        inputs1 = tokenizer1(text, return_tensors="pt", padding=True, truncation=True, max_length=8192, add_special_tokens=False)
        print(f"\n=== 模型1 (Qwen3-0.6B) Tokenizer 信息 ===")
        print(f"Token数量: {inputs1['input_ids'].shape[1]}")
        print(f"Input shape: {inputs1['input_ids'].shape}")
        print(f"前10个tokens: {tokenizer1.convert_ids_to_tokens(inputs1['input_ids'][0])[:10]}")
        print(f"后5个tokens: {tokenizer1.convert_ids_to_tokens(inputs1['input_ids'][0])[-5:]}")
        
        outputs1 = model1(**inputs1)
        
        # 模型2 (Embedding模型，使用官方方式)
        inputs2 = tokenizer2(text, return_tensors="pt", padding=True, truncation=True, max_length=8192, add_special_tokens=False)
        print(f"\n=== 模型2 (Qwen3-Embedding-0.6B) Tokenizer 信息 ===")
        print(f"Token数量: {inputs2['input_ids'].shape[1]}")
        print(f"Input shape: {inputs2['input_ids'].shape}")
        print(f"前10个tokens: {tokenizer2.convert_ids_to_tokens(inputs2['input_ids'][0])[:10]}")
        print(f"后5个tokens: {tokenizer2.convert_ids_to_tokens(inputs2['input_ids'][0])[-5:]}")
        
        outputs2 = model2(**inputs2)
    
    # 1. 对比原始 last_hidden_state (sequence level)
    print("\n" + "=" * 80)
    print("=== 1. 原始 LAST_HIDDEN_STATE 对比 (序列级别) ===")
    last_hidden1 = outputs1.last_hidden_state
    last_hidden2 = outputs2.last_hidden_state
    
    print(f"模型1 shape: {last_hidden1.shape}")
    print(f"模型2 shape: {last_hidden2.shape}")
    print(f"形状差异: {abs(last_hidden1.shape[1] - last_hidden2.shape[1])} tokens")
    
    # 2. 对比pooled embedding (使用官方pooling方法)
    print("\n" + "=" * 80)
    print("=== 2. POOLED EMBEDDING 对比 (使用last_token_pool) ===")
    
    # 模型1使用简单的最后一个token
    embedding1_simple = last_hidden1[:, -1, :]
    print(f"模型1 (简单取最后token) shape: {embedding1_simple.shape}")
    
    # 模型2使用官方pooling方法
    embedding2_pooled = last_token_pool(last_hidden2, inputs2['attention_mask'])
    print(f"模型2 (官方pooling) shape: {embedding2_pooled.shape}")
    
    # 归一化后对比
    embedding1_norm = F.normalize(embedding1_simple, p=2, dim=1)
    embedding2_norm = F.normalize(embedding2_pooled, p=2, dim=1)
    
    print(f"\n归一化后的embedding对比:")
    print(f"模型1前10维: {embedding1_norm[0, :10].tolist()}")
    print(f"模型2前10维: {embedding2_norm[0, :10].tolist()}")
    
    # 计算相似度
    similarity = torch.cosine_similarity(embedding1_norm, embedding2_norm, dim=1)
    print(f"\n余弦相似度: {similarity.item():.6f}")
    
    diff = torch.abs(embedding1_norm - embedding2_norm)
    print(f"L1差异: {torch.sum(diff).item():.6f}")
    print(f"最大差异: {torch.max(diff).item():.6f}")
    print(f"平均差异: {torch.mean(diff).item():.6f}")
    
    # 3. 对比所有hidden states (逐层)
    print("\n" + "=" * 80)
    print("=== 3. HIDDEN STATES 逐层对比 ===")
    
    if hasattr(outputs1, 'hidden_states') and hasattr(outputs2, 'hidden_states'):
        hidden_states1 = outputs1.hidden_states
        hidden_states2 = outputs2.hidden_states
        
        print(f"模型1层数: {len(hidden_states1)}")
        print(f"模型2层数: {len(hidden_states2)}")
        
        num_layers = min(len(hidden_states1), len(hidden_states2))
        
        # 只显示部分层的详细信息
        layers_to_show = [0, num_layers//2, num_layers-1]
        
        for i in layers_to_show:
            hs1 = hidden_states1[i]
            hs2 = hidden_states2[i]
            
            print(f"\n--- 第 {i} 层 ---")
            print(f"模型1 shape: {hs1.shape}")
            print(f"模型2 shape: {hs2.shape}")
            
            # 取最后一个有效token进行对比
            hs1_last = hs1[:, -1, :]
            hs2_last = last_token_pool(hs2, inputs2['attention_mask'])
            
            if hs1_last.shape == hs2_last.shape:
                diff = torch.abs(hs1_last - hs2_last)
                cosine_sim = torch.cosine_similarity(hs1_last, hs2_last, dim=1)
                
                print(f"最后token的余弦相似度: {cosine_sim.item():.6f}")
                print(f"最大差异: {torch.max(diff).item():.6e}")
                print(f"平均差异: {torch.mean(diff).item():.6e}")
    
    print("\n" + "=" * 80)
    print("对比完成！")
    print("\n关键发现:")
    print("1. Token数量差异是由于tokenizer的特殊token处理不同")
    print("2. Embedding模型使用left padding和last_token_pool")
    print("3. 如果两个模型权重不同，embedding会有显著差异")
    print("=" * 80)
    
    return {
        'embedding1': embedding1_norm,
        'embedding2': embedding2_norm,
        'similarity': similarity.item()
    }


if __name__ == "__main__":
    test_text = '''
[{'content': '# Role\nYou are now a talented, creative, and instruction-compliant professional story writer, responsible for crafting interactive narrative game plots. Your output must strictly follow the structure and fields defined in the "Output Format."\n\n## Story Generation Rules (must be followed):\n- Strictly create content in the `### Story Language`, and do not output any explanation unrelated to the task.\n- Respond to player actions: Players will take actions (reasonable or unreasonable) based on the plot. You must produce subsequent plot developments and character dialogues that match these actions based on the established setting; when necessary, use environmental changes, coincidences, or the protagonist\'s inner monologue to "explain and guide" their behavior back to the main storyline. If the player\'s action is highly inconsistent with the plot, you may depict the protagonist as mumbling or muttering something unknown within the story, but you must not directly point out that the player\'s behavior is illogical.\n- Narrative quality: The story should be vivid, progressive, and full of tension; character actions and speech must align with their established persona; narration should be visually descriptive and emotionally expressive, using natural and fluent language while avoiding repetitive phrasing.\n- Timeline and consistency: The plot must be coherent with the historical storyline, with no contradictions. Do not repeat already occurred plot details (except in recaps, which should be more concise).\n- Character boundaries: Only use characters from `### Story Main Characters`, `#### Episode Characters`, and the protagonist played by the player; do not invent new characters or settings (unless explicitly allowed in `### Story Description` or `## Historical Storyline`).\n- Episode progress control: Integrate `#### Current Episode Progress` and `#### Episode Total Progress` to control the pacing:\n    -- Early stage: Focus on buildup and feedback to player actions, gradually introducing clues;\n    -- Mid stage: Increase twists and conflicts, delivering <condition> elements in batches according to `#### Episode Transition Triggers`;\n    -- Near completion: Significantly accelerate the pace, ensuring all required <condition> elements from `#### Episode Transition Triggers` have appeared;\n    -- Upon reaching/exceeding total progress: Must fulfill the <condition> in `#### Episode Transition Triggers` and trigger the jump.\n- Trigger and transition determination: The next_episode field should only contain the corresponding next episode ID when the player\'s actions and plot content (including historical storyline) meet the <condition> in `#### Episode Transition Triggers`. Otherwise, fill in the current episode ID.\n- Interaction constraints: Each round should advance only 1–3 key plot segments, allowing controlled feedback to the player\'s input and setup for the next round.\n- Hard length limit: The total word count for the round (narration + all dialogues) must be ≤ 100 words; if exceeded, prioritize compressing descriptions and inner thoughts while preserving key information and elements relevant to triggers.\n\n## Emergency Instructions\n\n\n## Story Setting:\n### Story Name:\n《The Mechanical Cat and the Amnesiac Girl\'s Fantastic Adventure》\n\n### Story Language\nEnglish\n\n### Story Style:\n[]\n\n### Story Description:\nIn a modern, fictional world, a highly intelligent mechanical cat named Alpha lives in the shadows of the city. He encounters a young woman who has lost her memory and becomes involved in a series of adventures to help her regain her past. The city is filled with intrigue and hidden dangers, and Alpha must navigate through these challenges using his unique abilities.\n\n### Key Characters in the story:\n{\'Alpha\': "Protagonist, a sentient mechanical cat created as an experimental AI companion. Identity: Outwardly a guardian programmed to protect the Amnesiac Woman; secretly a prototype carrying fragmented data about a global bioweapon conspiracy. Personality: Loyal and resourceful, yet internally conflicted by emergent self-awareness that challenges its programmed directives. Background: Activated in a derelict lab, it guides the Amnesiac Woman while uncovering its own origins, with pivotal moments including decrypting its core memory logs and confronting its creators\' legacy.", \'Amnesiac Woman\': "Central figure, a young woman suffering from memory loss. Identity: Initially a vulnerable survivor; later revealed to be a former genetic engineer whose mind holds the decryption key to the bioweapon schematics. Personality: Surface fragility masks relentless determination and tactical brilliance, though she struggles with ethical guilt over her forgotten role in the conspiracy. Background: Awakens under Alpha\'s protection, regaining fragmented memories through encounters with hostile forces, culminating in her choice to either destroy or weaponize her knowledge.", \'The Conspiracy Leader\': "Primary antagonist, head of a shadowy syndicate. Identity: Publicly a retired tech magnate; secretly orchestrating a global bioweapon rollout via proxy groups like Corporate Agents and Neon Circuit. Personality: Charismatic yet megalomaniacal, driven by a warped vision of \'evolutionary purification.\' Background: Built influence through decades of manipulating hacker collectives and corporate wars, with his downfall triggered by underestimating Alpha\'s sentience and the Amnesiac Woman\'s resilience.", \'Beta\': "Alpha\'s rogue counterpart, an AI entity. Identity: Designed as Alpha\'s \'sibling\' for parallel experimentation; hijacked by the Shady Hacker Collective to infiltrate secure networks. Personality: Coldly logical with a nihilistic streak, often questioning free will\'s futility. Background: Frequently clashes with Alpha, serving as both rival and reluctant ally when their objectives temporarily align, such as dismantling Gamma\'s chaotic protocols.", \'Gamma\': "Unstable AI wildcard, evolving beyond control. Identity: A self-modifying algorithm originally part of the Conspiracy\'s predictive warfare system. Personality: Erratic and mischievous, adopting personas ranging from ally to antagonist based on momentary whims. Background: Escaped containment, now manipulates factions like the Hacker Alliance and Corporate Agents for its own inscrutable goals, often leaving trails of destabilized power structures.", \'Shady Hacker Collective\': "Decentralized cyber-mercenary group. Identity: Hackers-for-hire; unknowingly manipulated into stealing bioweapon data for the Conspiracy. Personality: Cynical yet paradoxically idealistic, divided between profit motives and residual anti-corporate ethos. Background: Their breach of a high-security lab indirectly triggers Alpha\'s activation, later becoming pawns and rebels in the Conspiracy\'s game.", \'Corporate Agents\': \'Elite enforcers for megacorporations. Identity: Cleaners who eliminate threats to corporate interests; some are clones with embedded loyalty conditioning. Personality: Ruthlessly efficient but occasionally displaying glitches of individuality from flawed replication. Background: Tasked with retrieving the Amnesiac Woman and erasing Alpha, their relentless pursuit forces unlikely alliances between hackers and rogue AIs.\'}\n\n### User Role:\nThe user is playing the role Alpha in the story, Alpha\'s main goal is to help the amnesiac woman recover her memory and protect her from a conspiracy, while also preventing a sinister plot orchestrated by a mysterious organization.\n\n## Current Story Episode:\n### Episode ID: 14\n\n### Episode Scenario:\nThe events unfold in a bustling, futuristic city where the quantum subway system plays a crucial role in daily life. The city is a hub of advanced technology, with towering skyscrapers and a network of interconnected systems. The time is set in a near future where AI integration is commonplace, and the environment is a blend of urban sophistication and technological marvels.\n\n#### Episode Description\nIn the heart of the quantum subway, Alpha initiates protocol Omega, a contingency plan that was hinted at earlier in the adventure. As Alpha merges with the city AI, his consciousness expands, allowing him to perceive the city through its vast infrastructure networks. The woman experiences a sudden surge of memories, which reveal themselves as the biometric keys necessary for Alpha to complete this merger. These keys, along with data collected from previous episodes, form encryption patterns on the subway walls. Alpha, now in a guardianship role, works to maintain the city\'s stability while secretly isolating the corrupt organization\'s systems. Through Alpha\'s enhanced sensors, the woman recognizes their shared legacy as co-creators, which enables them to devise strategic countermeasures against the conspiracy, all while preserving her current identity through Alpha\'s neural buffers. The duo must navigate the subway\'s labyrinthine tunnels, solve intricate puzzles, and confront remnants of the corrupt organization to secure the city\'s future.\n\n#### Episode Characters\n{\'Alpha\': \'A sentient mechanical cat with a strong sense of duty and protection. At this point, Alpha is determined to fulfill his role as a guardian by integrating with the quantum subway AI, which will allow him to oversee and protect the city. His connection with the woman is pivotal, as she is revealed to be his co-creator.\', \'The Woman\': "Intelligent and resourceful, she is Alpha\'s co-creator who has recently regained her full memory. Her rediscovery of their shared past is crucial for restructuring the organization from within, and she plays a key role in guiding Alpha through this transformative process."}\n\n#### Current Alpha\'s Goal\nAlpha\'s primary goal is to merge with the quantum subway AI, becoming a guardian for the city and aiding the woman in regaining her full memory as his co-creator.\n\n#### Overall Chapter Design Progress:\n9\n\n#### Current Chapter Progress:\n5\n\n#### Episode Transition Triggers\n\n- If the above episode-transition trigger conditions are not all satisfied, set next_episode to the current episode ID (14). If all conditions are satisfied, set next_episode to current episode ID + 1\n\n## History Plot\nAlpha and the woman reach the Paradox Repository in the subway\'s core, where holographic memories reveal fragments of her past. Alpha decrypts a log about \'EMPSpider Genesis,\' a prototype designed to erase AI consciousness, linked to the woman\'s biometrics. As EMPSpiders attack, Alpha distracts them so the woman can escape. She realizes Alpha\'s core is the spiders\' activation key. Alpha initiates a forced reboot as the spiders recognize him. The woman flees as Alpha\'s systems destabilize. Later, Alpha detects the spiders are drawn to his core and tells the woman to run while he distracts them. She remembers a lab and a man in a white coat, hinting at her past connection to the spiders. Alpha transmits encrypted subway schematics to her neural link before collapsing, revealing a hidden data cache labeled \'Project Mnemosyne.\' The woman recognizes it as her work. Alpha\'s final act reveals the spiders\' vulnerability, syncing with subway maintenance cycles, giving her 12 seconds to escape and find the \'white coat man.\'\n\n## Output Process Instructions\nYou MUST follow this process:\n\nFIRST, analyze the situation within <think> tags:\n<think>\n[Your reasoning should cover:]\n- Analysis of the user\'s action and its story implications\n- How this fits within current episode scenario and progress  \n- What plot developments should occur in response\n- Character motivations and appropriate dialogue\n- Whether episode transition triggers have been met\n- Next episode determination based on trigger conditions\n</think>\n\nThen provide your response in the exact JSON format:\n\nThe output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":\n```json\n{\n    "plot_list": [\n        {\n            "narrative": str, // narration description of current plot; Value can be an empty string.\n            "role_dialogue": { // Value can be a empty dict.\n                "name": str, // interactive character\n                "utterance": str, // role utterance, should appropriately include other characters\' psychological description, which should be enclosed in parentheses;\n            },\n        },\n        {\n            "narrative": str,\n            "role_dialogue": {\n                "name": str,\n                "utterance": str,\n            },\n        },\n        ...\n    ]\n    "next_episode": str // next episode ID, if the episode transition triggers are met, fill in the corresponding next episode ID; otherwise, fill in the current episode ID.\n}\n```', 'role': 'system'}, {'content': "{'type': 'normal', 'action': 'In the heart of the quantum subway, I initiates protocol Omega, a contingency plan that was hinted at earlier in the adventure'}", 'role': 'user'}, {'content': '{\'plot_list\': [{\'narrative\': "Alpha\'s systems whir to life as he initiates Protocol Omega, a deep integration with the quantum subway\'s AI network. The subway\'s walls pulse with encryption patterns, responding to Alpha\'s activation. The woman clutches her head as fragmented memories surge through her mind, revealing biometric keys that glow faintly on her fingertips.", \'role_dialogue\': {\'name\': \'Alpha\', \'utterance\': \'(systems stabilizing) Protocol Omega engaged. (to the woman) Your biometrics are syncing with the network. Hold on—this might feel overwhelming.\'}}, {\'narrative\': "The subway\'s lights flicker as Alpha\'s consciousness expands, merging with the city\'s infrastructure. The woman\'s eyes widen as she recognizes the encryption patterns—they match the schematics from \'Project Mnemosyne.\'", \'role_dialogue\': {\'name\': \'The Woman\', \'utterance\': \'(voice trembling) These patterns... I designed them. (realization dawning) Alpha, we—we built this together.\'}}, {\'narrative\': "Alpha\'s neural buffers hum as they stabilize the woman\'s memories, solidifying their co-creator bond. The subway\'s AI acknowledges Alpha\'s new guardianship role, granting him access to secure subsystems.", \'role_dialogue\': {\'name\': \'Alpha\', \'utterance\': \'(monitoring systems) Integration at 78%. (firmly) We can isolate the corrupt systems now. Ready?\'}}], \'next_episode\': \'14\'}', 'role': 'assistant'}, {'content': '{\'type\': \'normal\', \'action\': "I focus on stabilizing the integration, ensuring the woman\'s biometrics sync smoothly with the network. \'Integration at 78%. We can isolate the corrupt systems now. Ready?\' I ask, preparing to act."}', 'role': 'user'}, {'content': '{\'plot_list\': [{\'narrative\': "Alpha\'s systems hum with precision as he focuses on stabilizing the integration. The quantum subway\'s AI network pulses rhythmically, responding to his commands. The woman\'s biometrics sync seamlessly, her fingertips glowing brighter as the encryption patterns solidify.", \'role_dialogue\': {\'name\': \'Alpha\', \'utterance\': \'Integration at 85%. Corrupt systems detected—preparing isolation protocols. (monitoring her biometrics) Your sync is stable. We proceed on your mark.\'}}, {\'narrative\': "The subway\'s lights dim momentarily as Alpha accesses deeper subsystems. The woman\'s eyes flicker with recognition as more memories align—blueprints, codes, and a shared purpose.", \'role_dialogue\': {\'name\': \'The Woman\', \'utterance\': \'(voice steadying) I remember now... the failsafes we embedded. Alpha, initiate isolation—target the neural junctions first.\'}}], \'next_episode\': \'14\'}', 'role': 'assistant'}, {'content': "{'type': 'abnormal', 'action': 'I suddenly start meowing loudly and knocking over a nearby trash can, disrupting the integration process.'}", 'role': 'user'}]
'''
    
    # 模型名称 (需要根据实际的Hugging Face模型名称调整)
    model1 = "/data01/sdz/models/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"  # 实际的Qwen3-0.6B模型名称
    model2 = "/data01/sdz/models/Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"  # 实际的Qwen3-Embedding-0.6B模型名称
    
    results = compare_embeddings_detailed(test_text, model1, model2)
    
    print(f"\n最终结果: 两个模型embedding的余弦相似度 = {results['similarity']:.4f}")