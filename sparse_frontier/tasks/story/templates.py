from typing import List, Optional, Set

class ItemGenerator:
    """Generates descriptive item names by sampling random attributes."""
    adjectives = [
        "ancient", "ornate", "engraved", "weathered", "gilded", "polished", "exquisite",
        "intricate", "delicate", "sturdy", "mystic", "legendary", "pristine", "refined",
        "timeworn", "ceremonial", "lavish", "forgotten"
    ]
    materials = [
        "bronze", "silver", "gold", "ivory", "jade", "emerald", "crystal", "obsidian",
        "ruby", "sapphire", "marble", "amber", "copper", "lapis", "porcelain"
    ]
    item_types = [
        "amulet", "figurine", "sword", "shield", "goblet", "necklace", "ring",
        "statuette", "pendant", "candelabra", "anklet", "comb", "seal", "chalice",
        "horn", "key", "bracelet", "mask", "lamp", "idol", "vase"
    ]

    @classmethod
    def sample_item(cls, random_obj) -> str:
        """Randomly generates an item name from adjectives, materials, and types."""
        adj = random_obj.choice(cls.adjectives)
        mat = random_obj.choice(cls.materials)
        it = random_obj.choice(cls.item_types)
        return f"{adj} {mat} {it}"
    
    @classmethod
    def _get_unique_item(cls, items_seen: Set[str], random_obj) -> str:
        for _ in range(1000):
            candidate = cls.sample_item(random_obj)
            if candidate not in items_seen:
                items_seen.add(candidate)
                return candidate

        raise ValueError("Could not find a unique item.")


class NarrativeResources:
    """Holds narrative building blocks."""
    CHARACTERS = [
        "Marcus", "Selene", "Cassius", "Daria", "Niko", "Iris", "Leon", "Marina", "Octavia", "Petros",
        "Lyra", "Thanos", "Eliana", "Hector", "Rhea", "Adrian", "Sophia", "Dorian", "Cleo", "Phaedra",
        "Lysander", "Callisto", "Evander", "Hermia", "Althea", "Brynn", "Cyrus", "Damon", "Eirene",
        "Felix", "Gaia", "Hylas", "Ianthe", "Jonas", "Kallias", "Leda", "Myrto", "Neon", "Orestes",
        "Phile", "Qarion", "Roxana", "Stavros", "Timon", "Urania", "Vitalis", "Xanthe", "Yanis", "Zarek"
    ]

    LOCATIONS = [
        "Alexandria", "Byzantium", "Rhodes", "Carthage", "Athens", "Rome",
        "Sparta", "Babylon", "Memphis", "Thebes", "Troy", "Delphi",
        "Ephesus", "Pergamon", "Syracuse", "Corinth", "Knossos", "Miletus",
        "Argos", "Olympia", "Phaselis", "Halicarnassus", "Gortyn", "Mycenae",
        "Antioch", "Palmyra", "Damascus", "Persepolis", "Susa", "Ecbatana",
        "Nineveh", "Ur", "Tyre", "Sidon", "Jerusalem", "Petra", "Cyrene",
        "Leptis Magna", "Volubilis", "Caesarea", "Nicaea", "Thessalonica",
        "Philippi", "Ravenna", "Massilia", "Londinium", "Lutetia", "Tarraco",
        "Emerita Augusta", "Tingis", "Hippo Regius", "Ctesiphon", "Pataliputra",
        "Taxila", "Merv", "Samarkand", "Bukhara", "Thonis-Heracleion",
        "Berenice", "Adulis", "Axum", "Meroe", "Napata", "Berenice Troglodytica"
    ]

    EVENTS = [
        "a grand festival", "a violent storm", "a secret treaty", "an unexpected betrayal",
        "a tense negotiation", "a mysterious disappearance", "a lavish wedding feast",
        "an outbreak of conflict", "a solemn religious ceremony", "a dangerous ambush",
        "a sudden market crash", "a surprising alliance", "a forbidden ritual",
        "an urgent council meeting", "a whispered rumor spreading", "an opulent banquet",
        "a chaotic uprising", "a quiet funeral procession", "a midnight duel", "a rare eclipse"
    ]

    SCENE_INTRO_TEMPLATES = [
        "The sun was low as {protagonist} arrived at {location}, the air filled with distant voices. ",
        "At dawn, {protagonist} reached the gates of {location}, where merchants and travelers converged. ",
        "As evening fell, {protagonist} entered {location}, its quiet streets hiding subtle mysteries. ",
        "{protagonist} approached {location} under a pale sky, senses attuned to every whisper. ",
        "With steady steps, {protagonist} crossed into {location}, drawn by distant murmurs. ",
        "Under fading daylight, {protagonist} set foot in {location}, eager to learn what it offered. ",
        "{protagonist} came upon {location}, where old walls and new faces stood in delicate balance. ",
        "Beneath gentle breezes, {protagonist} ventured into {location}, curious about its secrets. ",
        "{location} greeted {protagonist} with silent promises and unknown challenges. ",
        "The threshold of {location} welcomed {protagonist}, who felt the weight of untold stories. "
    ]

    REASON_TEMPLATES = [
        "{protagonist} came here hoping to learn something new, or perhaps gain an advantage. ",
        "This place might hold a clue {protagonist} had long sought. ",
        "Weary from past travels, {protagonist} searched {location} for insight or respite. ",
        "{protagonist} believed that {location} hid key knowledge beneath its calm facade. ",
        "Rumors had guided {protagonist} here, hinting at treasures of wisdom or wealth. ",
        "Long journeys had led {protagonist} to {location}, a step closer to understanding. ",
        "In pursuit of truth, {protagonist} looked to {location} for subtle revelations. ",
        "{protagonist} trusted that {location} might provide the missing piece of a larger puzzle. ",
        "A quiet determination brought {protagonist} to {location}, ever searching for meaning. ",
        "Instinct told {protagonist} that {location} would reveal something of value. "
    ]

    EVENT_TEMPLATES = [
        "Not long after arriving, {event} shook the local order. ",
        "Soon enough, {event} seized everyone's attention. ",
        "Within hours, {event} disrupted the familiar routines. ",
        "It was not long before {event} set the stage for new tensions. ",
        "Before {protagonist} could settle, {event} altered the mood of the streets. ",
        "Amid cautious steps, {event} became the talk of every corner. ",
        "Hardly had {protagonist} arrived before {event} stirred uneasy whispers. ",
        "{event} cast its shadow over {location}, changing plans and minds. ",
        "Like a ripple in still water, {event} spread through {location}. ",
        "The calm surface broke as {event} brought both fear and opportunity. "
    ]

    CHAR_INTRO_TEMPLATES = [
        "There, {protagonist} encountered {character}, who seemed eager to exchange words or goods. ",
        "In a quiet courtyard, {protagonist} met {character}, a figure with subtle ambitions. ",
        "{character} approached {protagonist}, eyes bright with opportunity. ",
        "As {protagonist} ventured further in, {character} introduced themselves with measured courtesy. ",
        "It was {character} who stepped forward, offering guidance or perhaps misdirection. ",
        "{character} emerged from the crowd, greeting {protagonist} with a knowing smile. ",
        "{protagonist} found {character} awaiting them, curious about their intentions. ",
        "{character} appeared as if expecting {protagonist}, engaging them without delay. ",
        "Among the gathered faces, {character} beckoned {protagonist} closer. ",
        "{protagonist} soon attracted {character}, who offered a few quiet words. "
    ]

    CONVERSATION_EXTENSION_TEMPLATES = [
        "In hushed tones, they spoke of local customs and distant rumors, sharing hints of hidden pathways. ",
        "The two discussed the shifting fortunes of traders and travelers, pondering unseen forces at play. ",
        "A short exchange revealed uncharted corners of {location}, where knowledge or secrets might dwell. ",
        "They lingered over tales of old alliances and forgotten disputes, weaving past into present. ",
        "Together, they reflected on the nature of trust and deceit, aware that fate often twists. ",
        "They debated the meaning of recent events, each seeking patterns in the chaos. ",
        "A brief conversation unveiled hints of rare opportunities that might lie just beyond reach. ",
        "Quiet words passed between them, unveiling fragments of a larger puzzle forming in {location}. ",
        "They considered old legends that might still echo in deserted alleys and abandoned shrines. ",
        "Their dialogue danced around subtle clues, each suggestion hinting at treasures undiscovered. ",
        "They whispered about shifting alliances within {location}, where even a friend's loyalty might be uncertain. ",
        "Subtle gestures hinted at undercurrents of tension, as hidden interests shaped every choice around them. ",
        "They exchanged quiet observations about the city's invisible networks, ties born of necessity and ambition. ",
        "Their words traced over delicate negotiations that had once sealed lasting truces in {location}. ",
        "They acknowledged lingering resentments harbored by those wronged in bygone eras, their stories still alive in whispers. ",
        "In a calm moment, they compared notes on the traders who passed through {location}, each leaving their subtle mark. ",
        "Their reflections turned to the interplay of supply and demand, seeing how fortunes might turn in an instant. ",
        "In measured tones, they recalled cryptic hints from travelers who dared to breach distant borders. ",
        "They speculated on the motives of silent watchers lurking in the shadows of {location}, ever present but unseen. ",
        "Carefully, they navigated the topic of old feuds, wary of awakening dormant animosities that still simmered. ",
        "They recounted whispers of clandestine markets hidden behind ordinary facades, where bargains could shift power. ",
        "Their conversation danced around unspoken rules governing respect and retribution, each boundary a subtle test. ",
        "They paused to consider the influence of distant rulers, whose far-off edicts still rippled through {location}. ",
        "Noting patterns in currency flows, they saw reflections of deeper unrest simmering beneath calm surfaces. ",
        "They delved into the subtle art of earning trust in a place where trust was scarce and hard-won. ",
        "They paused, reflecting on the quiet desperation that drove many to risk everything for a chance at change. ",
        "Their words lingered on rumors of distant lands, where fortunes or ruin awaited bold seekers. ",
        "They compared accounts of strange visitors bearing knowledge or confusion, each arrival a new riddle in {location}. ",
        "They lingered over accounts of hidden sanctuaries, hushed refuges that offered shelter from the chaos outside. ",
        "In the end, they acknowledged that truly understanding {location} required patience, insight, and steady resolve. "
    ]

    BUYING_TEMPLATES = [
        "After careful negotiation, {protagonist} acquired {item} from {character}. ",
        "With measured consideration, {protagonist} purchased {item} from {character}, examining it closely. ",
        "{protagonist} exchanged words and coin with {character} to secure {item}. ",
        "Seeing value in {character}'s offer, {protagonist} obtained {item}. ",
        "{item} changed hands as {protagonist} completed the purchase from {character}. ",
        "Following subtle bargaining with {character}, {protagonist} claimed ownership of {item}. ",
        "Weighing {character}'s worth carefully, {protagonist} bought {item}. ",
        "The transaction concluded with {protagonist} acquiring {item} from {character}. ",
        "After reaching terms with {character}, {protagonist} took possession of {item}. "
    ]

    SELLING_TEMPLATES = [
        "With practiced ease, {protagonist} sold {item} to {character}. ",
        "{protagonist} completed the sale of {item} to {character}. ",
        "Through skilled negotiation, {protagonist} traded {item} to {character}. ",
        "{protagonist} transferred {item} to {character}, both parties satisfied. ",
        "{protagonist} sold {item} to {character} at fair value. ",
    ]

    FAREWELL_TEMPLATES = [
        "{protagonist} bid {character} a measured farewell before continuing onward. ",
        "With a final nod, {protagonist} parted ways with {character}. ",
        "In quiet understanding, {protagonist} left {character}, their paths diverging. ",
        "{protagonist} offered a calm goodbye to {character} before departing. ",
        "{protagonist} turned from {character}, ready to move on. ",
        "{protagonist} spoke a brief farewell to {character} before leaving. ",
        "{protagonist} and {character} exchanged polite goodbyes. ",
        "{protagonist} ended their meeting with {character} with a subtle parting glance. ",
        "No more words were needed; {protagonist} stepped away from {character}. ",
        "With a light gesture, {protagonist} acknowledged {character} once more before departing. "
    ]

    CONCLUSION_TEMPLATES = [
        "Eventually, {protagonist} moved on, carrying new impressions forward. ",
        "As {protagonist} prepared to depart, the path ahead remained uncertain but compelling. ",
        "The chapter closed as {protagonist} stepped away, leaving {location} to its own mysteries. ",
        "With quiet resolve, {protagonist} carried these moments onward into the unknown. ",
        "In time, {protagonist} slipped away, intent on shaping what they had gleaned. ",
        "As the day waned, {protagonist} held new pieces of a grander puzzle. ",
        "Departing, {protagonist} carried fresh insights wrapped in lingering questions. ",
        "With each step, {protagonist} bore away a subtle shift in perspective. ",
        "Nothing would be the same as {protagonist} left {location}, thoughts turning inward. ",
        "In parting, {protagonist} acknowledged that the journey still had far to run. "
    ]

    EXTRA_TEMPLATES = [
        "Shadows stretched long, hinting at stories yet to unfold. ",
        "The distant hum of voices hinted at unseen deals. ",
        "Hidden corners of the city promised knowledge or peril. ",
        "Old stones bore silent witness to countless secrets. ",
        "A subtle tension lingered, as though fate held its breath. ",
        "Each new face carried hints of past encounters unspoken. ",
        "In quiet corners, ambitions simmered, waiting for a spark. ",
        "The taste of mystery clung to the air, thick and potent. ",
        "Somewhere, a whisper promised answers for those who dared. ",
        "Lingering dusk brought gentle uncertainty to weary minds. "
    ]

    @staticmethod
    def choose(templates: List[str], random_obj, protagonist: str, location: Optional[str] = None,
               event: Optional[str] = None, character: Optional[str] = None, item: Optional[str] = None) -> str:
        template = random_obj.choice(templates)
        return template.format(protagonist=protagonist, location=location, event=event, character=character, item=item)

    @staticmethod
    def choose_multiple(templates: List[str], n: int, random_obj, protagonist: str, location: Optional[str] = None,
                        event: Optional[str] = None, character: Optional[str] = None, item: Optional[str] = None) -> List[str]:
        selected = random_obj.sample(templates, n)
        return [
            t.format(protagonist=protagonist, location=location, event=event, character=character, item=item)
            for t in selected
        ]

TASK_INTRO = (
    "You are given a narrative composed of multiple chapters. Throughout these chapters, "
    "the protagonist travels between different locations, meets various characters, and "
    "engages in trading activities. All items mentioned in the "
    "narrative are unique, and their ownership can change through trades. Your task is to "
    "carefully read the narrative and answer the questions based on the provided information."
)
