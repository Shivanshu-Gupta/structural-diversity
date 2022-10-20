import re

from generation.scfg.generate import Generator

if 0:
    def anonymize_covr_target(target):
        anonymized = target
        numbers = ['2', '3', '4']
        anonymized = re.sub(f"(\\b)({'|'.join(numbers)})(\\b)", r'ANON_NUMBER', anonymized)
        entities = ['dog', 'cat', 'mouse', 'animal']
        anonymized = re.sub(f"(\\b)({'|'.join(entities)})(\\b)", r'ANON_ENTITY', anonymized)
        relations = ['chasing', 'playing with', 'looking at']
        anonymized = re.sub(f"(\\b)({'|'.join(relations)})(\\b)", r'ANON_RELATION', anonymized)
        types = ['color', 'shape']
        anonymized = re.sub(f"(\\b)({'|'.join(types)})(\\b)", r'ANON_TYPE', anonymized)
        types_values = ['black', 'white', 'brown', 'gray', 'round', 'square', 'triangle']
        anonymized = re.sub(f"(\\b)({'|'.join(types_values)})(\\b)", r'ANON_TYPE_VALUE', anonymized)
        symbols = ['or', 'and']
        anonymized = re.sub(f"(\\b)({'|'.join(symbols)})(\\b)", r'ANON_LOGIC', anonymized)
        return anonymized
else:
    def anonymize_covr_target(target):
        anonymized = target
        entities = ['2', '3', '4', 'dog', 'cat', 'mouse', 'animal', 'chasing', 'playing with', 'looking at', 'color',
                    'shape', 'black', 'white', 'brown', 'gray', 'round', 'square', 'triangle']
        anonymized = re.sub(f"(\\b)({'|'.join(entities)})(\\b)", r'ANON_ENTITY', anonymized)
        symbols = ['or', 'and']
        anonymized = re.sub(f"(\\b)({'|'.join(symbols)})(\\b)", r'ANON_SYMBOL', anonymized)
        return anonymized

def anonymize_covr_target_using_prods(productions):
    anonymized = Generator.generate_from_prods(productions, template=True)
    return anonymized

def anonymize_overnight_target(target):
    anonymized = target
    anonymized = re.sub(r'string [^)]*', 'ANON_STRING', anonymized)
    anonymized = re.sub(r'en\.[^ )]*', 'ANON_ENTITY', anonymized)
    anonymized = re.sub(r'number [^)]*', 'ANON_NUMBER', anonymized)
    return anonymized


def anonymize_atis_target(target):
    anonymized = re.sub(r'(^|\W)\w+ :', r'\1ANON_ENTITY :', target)
    return anonymized


def anonymize_thingtalk(target):
    anonymized = ' '.join(target.split())
    fields = ['alumniOf', 'Person', 'award', 'knowsLanguage', 'address.addressLocality', 'email', 'url', 'jobTitle',
              'image', 'id', 'worksFor', 'workLocation', 'telephone', 'faxNumber', 'location']
    for field in fields:
        anonymized = anonymized.replace(field, 'ANON')
    fields = ["==", ">=", "<=", "~=", "=~", "contains~", "contains", "asc", "desc"]
    for field in fields:
        anonymized = anonymized.replace(field, 'SYMBOL')
    anonymized = anonymized.replace("QUOTED_ VAL", "QUOTED_VAL")
    anonymized = anonymized.replace("ANON: QUOTED_VAL", "QUOTED_VAL")
    anonymized = anonymized.replace("ANON:VAL", "QUOTED_VAL")
    anonymized = anonymized.replace("ANON: VAL", "QUOTED_VAL")
    anonymized = anonymized.replace("[", "(")
    anonymized = anonymized.replace("]", ")")
    return anonymized

def anonymize_smcalflow_target(target, anonymize_level=1):
    anonymized = target
    if anonymize_level > 0:
        anonymized = re.sub(r'".+?"', 'ANON_STRING', anonymized)
    if anonymize_level > 0.5:
        anonymized = re.sub(r'\d+?L', r'ANON_NUMBER', anonymized)
        anonymized = re.sub(r' \d+?([ |\)])', r' ANON_NUMBER\1', anonymized)
    if anonymize_level > 1:
        anonymized = re.sub(r'([A-z]+?)\.([A-z]+?)', r'\1.ANON_PROPERTY', anonymized)
        anonymized = re.sub(r'([A-z]+?)\.([A-z]+?)', r'\1.ANON_PROPERTY', anonymized)
    return anonymized
