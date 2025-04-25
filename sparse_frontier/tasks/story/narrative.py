from typing import List, Optional, Set, Tuple

from sparse_frontier.tasks.story.templates import NarrativeResources, ItemGenerator


class Chapter:
    """Represents a single narrative chapter."""
    def __init__(self, chapter_id: int, location: str, character: str, event: str, random_obj):
        self.chapter_id = chapter_id
        self.location = location
        self.event = event
        self.character = character
        self.random_obj = random_obj
        self.structure = {
            'scene_introduction': [],
            'encounter': [],
            'conversation_extension': [],
            'buying_transactions': [],
            'selling_transactions': [],
            'farewell': [],
            'scene_conclusions': [],
            'extra': []
        }
        self.bought_items: List[str] = []
        self.sold_items: List[str] = []
        self.end_item_count: Optional[int] = None

    def compile_text(self) -> str:
        parts = (
            self.structure['scene_introduction']
            + self.structure['encounter']
            + self.structure['conversation_extension']
            + self.structure['buying_transactions']
            + self.structure['selling_transactions']
            + self.structure['farewell']
            + self.structure['scene_conclusions']
            + self.structure['extra']
        )
        return f"Chapter {self.chapter_id}:\n" + "".join(parts)

    def generate_chapter(self, protagonist_name: str, conversation_extensions: int, items_seen: Set[str], inventory: List[str]) -> None:
        """Generates a complete chapter with all narrative elements."""
        # Scene introduction
        self.structure['scene_introduction'].append(
            NarrativeResources.choose(NarrativeResources.SCENE_INTRO_TEMPLATES,
                                    protagonist=protagonist_name, location=self.location, random_obj=self.random_obj)
        )
        self.structure['scene_introduction'].append(
            NarrativeResources.choose(NarrativeResources.REASON_TEMPLATES,
                                    protagonist=protagonist_name, location=self.location, random_obj=self.random_obj)
        )
        self.structure['scene_introduction'].append(
            NarrativeResources.choose(NarrativeResources.EVENT_TEMPLATES,
                                    protagonist=protagonist_name, event=self.event, location=self.location, random_obj=self.random_obj)
        )

        # Encounter
        self.structure['encounter'].append(
            NarrativeResources.choose(NarrativeResources.CHAR_INTRO_TEMPLATES,
                                    protagonist=protagonist_name, character=self.character, random_obj=self.random_obj)
        )

        # Conversation extensions
        self.structure['conversation_extension'].extend(
            NarrativeResources.choose_multiple(
                NarrativeResources.CONVERSATION_EXTENSION_TEMPLATES,
                conversation_extensions,
                protagonist=protagonist_name,
                location=self.location,
                random_obj=self.random_obj
            )
        )

        # Handle buying and selling
        new_item = ItemGenerator._get_unique_item(items_seen, self.random_obj)
        inventory.append(new_item)
        self.bought_items.append(new_item)
        self.structure['buying_transactions'].append(
            NarrativeResources.choose(NarrativeResources.BUYING_TEMPLATES,
                                    protagonist=protagonist_name,
                                    character=self.character,
                                    item=new_item,
                                    random_obj=self.random_obj)
        )

        older_items = [it for it in inventory if it != new_item]
        if older_items and self.random_obj.random() < 0.5:
            sell_item = self.random_obj.choice(older_items)
            inventory.remove(sell_item)
            self.sold_items.append(sell_item)
            self.structure['selling_transactions'].append(
                NarrativeResources.choose(NarrativeResources.SELLING_TEMPLATES,
                                        protagonist=protagonist_name,
                                        character=self.character,
                                        item=sell_item,
                                        random_obj=self.random_obj)
            )
        
        self.end_item_count = len(inventory)

        # Farewell
        self.structure['farewell'].append(
            NarrativeResources.choose(NarrativeResources.FAREWELL_TEMPLATES,
                                    protagonist=protagonist_name, character=self.character, random_obj=self.random_obj)
        )

        # Scene conclusions
        self.structure['scene_conclusions'].append(
            NarrativeResources.choose(NarrativeResources.CONCLUSION_TEMPLATES,
                                    protagonist=protagonist_name, location=self.location, random_obj=self.random_obj)
        )

        # Add extra sentences
        self.structure['extra'].extend(
            NarrativeResources.choose_multiple(NarrativeResources.EXTRA_TEMPLATES, 1,
                                             protagonist=protagonist_name, location=self.location, random_obj=self.random_obj)
        )


class NarrativeGenerator:
    """Generates a base narrative with a certain number of chapters."""
    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        random_obj,
        protagonist_name: str = "Arion",
        conversation_extensions: int = 3,
    ):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.protagonist_name = protagonist_name
        self.protagonist_inventory: List[str] = []
        self.conversation_extensions = conversation_extensions
        self._used_pairs: Set[Tuple[str, str]] = set()
        self._items_seen: Set[str] = set()
        self.random_obj = random_obj

        self.chapters = self._generate_base_narrative()
        assert len(self.tokenizer.text_to_tokens(self.compile_narrative())) <= self.sequence_length, "Narrative is too long."
    
    def compile_narrative(self) -> str:
        return "\n\n".join(chapter.compile_text() for chapter in self.chapters)

    def _generate_base_narrative(self) -> List[Chapter]:
        chapters = []
        total_tokens = 0
        chapter_id = 1

        while True:
            assert chapter_id <= 2000, "Limit reached."
            for _ in range(100):
                location = self.random_obj.choice(NarrativeResources.LOCATIONS)
                character = self.random_obj.choice(NarrativeResources.CHARACTERS)
                if (location, character) not in self._used_pairs:
                    self._used_pairs.add((location, character))
                    break
            else:
                raise ValueError("Could not find a unique (location, character) pair.")

            event = self.random_obj.choice(NarrativeResources.EVENTS)
            chapter = Chapter(chapter_id, location, character, event, self.random_obj)

            chapter.generate_chapter(
                self.protagonist_name,
                self.conversation_extensions,
                self._items_seen,
                self.protagonist_inventory
            )

            chapter_text = chapter.compile_text()
            chapter_tokens = len(self.tokenizer.text_to_tokens(f'\n\n{chapter_text}'))

            if total_tokens + chapter_tokens > self.sequence_length:
                break

            chapters.append(chapter)
            total_tokens += chapter_tokens
            chapter_id += 1

        return chapters
