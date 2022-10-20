import editdistance
import re

TOKENS_TO_IGNORE = ["(", ")", ","]


def tokenize_lf(lf, add_sos=True):
    target = lf.replace(' [ ', '[').replace(' ]', ']').replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
    if add_sos:
        target = f"<s> ( {target} )"
    tokens = target.split()
    return tokens


def jaccard(ngrams_with_same_first, ngrams_with_similar_token_as_first):
    inter = ngrams_with_same_first.intersection(ngrams_with_similar_token_as_first)
    uni = ngrams_with_same_first.union(ngrams_with_similar_token_as_first)
    return len(inter) / len(uni)


def normalize_target_string(target_string):
    return target_string.replace(' ,', ',').replace(', ', ',').replace('string!', 'string !').replace('call.', 'call .')


clauses_regex = clauses_regex = re.compile(r"(?:(?<=and )|(?<=or )|(?<=\) ))\( .*?VAL \]? ?\)")
def normalize_thingtalk_and_clauses(target):
    # normalize and/or clauses by alphabet
    clauses = clauses_regex.findall(target)

    if not clauses:
        return target

    and_or = re.findall(" (and|or) ", target)
    if len(and_or) + 1 != len(clauses):
        return target

    # we assume string was already normalized such that "or" clauses appear first.
    # since or clauses are computed first in thingtalk, we only normalize the "and" clauses that appear after "or"
    n_or = len([c for c in and_or if c == "or"])
    if n_or > 0:
        assert "or" not in and_or[n_or+1:]
        new_clauses = clauses[:n_or+1] + sorted(clauses[n_or+1:])
    else:
        new_clauses = sorted(clauses)

    assert len(new_clauses) == len(clauses)

    # re-order the clauses by masking them then reassigning
    for old_clause in clauses:
        target = target.replace(old_clause, "!REPLACE!", 1)
    if target.count("!REPLACE!") != len(new_clauses):
        return target
    for new_clause in new_clauses:
        target = target.replace("!REPLACE!", new_clause, 1)
    return target


def normalize_thingtalk(target):
    # there are two correct ways to count a field used in the thingtalk data - since both are correct and seen in data we
    # should take this into account in evaluation
    match_string = r"compute count \( (\w+) \) of \( \( sort count (asc|desc) of \( compute count \( \w+ \) of"
    count_case_relevant = re.findall(match_string, target)
    if count_case_relevant:
        target = re.sub(match_string, r"( sort count \2 of ( compute count ( \1 ) of", target)
        target = target.replace("[ NUMBER_VAL ] )", "[ NUMBER_VAL ]")

    # we evaluate exact string match but the order of multiple and/or clauses shouldn't matter for evaluation. We
    # normalize such that the "or" clause (which weirdly is computed in this language before "and") will always appear
    # first
    for _ in range(2):
        # do this multiple times since each time moves the "or" clause left
        target = re.sub(r"(\( [^\)]*? \)) (and) (\( [^\)]*? \) or \( [^\)]*? \))", r"\3 and \1", target)

    target = normalize_thingtalk_and_clauses(target)

    return target


def check_error_at_unobserved(gold, prediction, unobserved_structures):
    gold_tokens = tokenize_lf(gold)
    pred_tokens = tokenize_lf(prediction)

    if any(len(us) != 2 for us in unobserved_structures):
        raise ValueError()

    i = 0
    while i < min(len(gold_tokens), len(pred_tokens)):
        if gold_tokens[i] != pred_tokens[i]:
            break
        i += 1
    else:
        return False

    gold_token_on_error = gold_tokens[i]
    for (a, b) in unobserved_structures:
        if gold_token_on_error == b and a in gold_tokens:
            return True

    return False


def check_targets_similar(target1, target2, dataset, allow_wrong_parentheses=0, lstm=True):
    # Remove all whitespaces and T5 special characters (which are not kept) to standardize the strings before
    # comparisons. This is not ideal but as far as we've seen, the probability of a false-positive because of this
    # removal is negligible.

    if lstm:
        # bunch of ugly fixes due to wrong post-processing in experiments which will require re-running everything...
        target1 = target1.replace(' ##', '').replace("<", "").replace("=", " = ").replace("-", " - ").replace(">", " > ").replace(":", " : ").replace("!", " ! ").replace("~", "").replace("^", "").replace("(", " ( ").replace(")", " ) ").replace("_", " _ ").replace(".", " ")
        target2 = target2.replace(' ##', '').replace("<", "").replace("=", " = ").replace("-", " - ").replace(">", " > ").replace(":", " : ").replace("!", " ! ").replace("~", "").replace("^", "").replace("(", " ( ").replace(")", " ) ").replace("_", " _ ").replace(".", " ")

        target1 = ' '.join(target1.split())  # normalize whitespaces
        target2 = ' '.join(target2.split())  # normalize whitespaces

        target1 = target1.replace(' _ ', '_').replace('quoted_val', 'QUOTED_VAL').replace('number_val', 'NUMBER_VAL').replace('val', 'VAL')
        target2 = target2.replace(' _ ', '_').replace('quoted_val', 'QUOTED_VAL').replace('number_val', 'NUMBER_VAL').replace('val', 'VAL')

    if dataset == "thingtalk":
        target1 = normalize_thingtalk(target1)
        target2 = normalize_thingtalk(target2)

    target1 = target1.replace(' ##', '').replace("<", "").replace("~", "").replace("^", "").replace(".", " ").lower()
    target2 = target2.replace(' ##', '').replace("<", "").replace("~", "").replace("^", "").replace(".", " ").lower()

    target1 = ' '.join(target1.split())  # normalize whitespaces
    target2 = ' '.join(target2.split())  # normalize whitespaces

    if dataset == "overnight":
        # temporary heuristic evaluation fix for overnight -
        tokens1 = sorted(tokenize_lf(target1))
        tokens2 = sorted(tokenize_lf(target2))
        target1 = ' '.join(tokens1)
        target2 = ' '.join(tokens2)

    target1 = target1.replace(" ", "")
    target2 = target2.replace(" ", "")

    if target1 == target2:
        return True
    elif allow_wrong_parentheses == 0:
        return False

    if target1.replace("(", "").replace(")", "") != target2.replace("(", "").replace(")", ""):
        return False

    parentheses1 = [c for c in target1 if c in ["(", ")"]]
    parentheses2 = [c for c in target2 if c in ["(", ")"]]

    return editdistance.eval(parentheses1, parentheses2) <= allow_wrong_parentheses


if __name__ == "__main__":
    print(check_targets_similar("hello((()))", "hello((()))", "thingtalk"))
    print(check_targets_similar("hello((()))", "hello((())", "thingtalk"))
    print(check_targets_similar("hello((()))", "hello((()", "thingtalk"))
    print(check_targets_similar("hello((()))", "hello(()", "thingtalk"))

    assert (
        check_targets_similar(
            "filter ( Person ) ( alumniOf contains QUOTED_VAL ) or ( alumniOf contains QUOTED_VAL ) and ( alumniOf contains QUOTED_VAL ) and ( workLocation == location: QUOTED_VAL )",
            "filter ( Person ) ( alumniOf contains QUOTED_VAL ) or ( alumniOf contains QUOTED_VAL ) and ( workLocation == location: QUOTED_VAL ) and ( alumniOf contains QUOTED_VAL )",
            "thingtalk"
        ) is True
    )

    assert (
        check_targets_similar(
            "filter ( Person ) ( alumniOf1 contains QUOTED_VAL ) and ( alumniOf2 contains QUOTED_VAL ) and ( workLocation == location: QUOTED_VAL )",
            "filter ( Person ) ( alumniOf2 contains QUOTED_VAL ) and ( workLocation == location: QUOTED_VAL ) and ( alumniOf1 contains QUOTED_VAL )",
            "thingtalk"
        ) is True
    )

