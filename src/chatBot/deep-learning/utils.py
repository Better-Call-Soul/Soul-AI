def map_tag_pattern(df, tag_col, text_col, res_col):
    tags = []
    inputs = []
    responses = {}

    for _, item in df.iterrows():
        ptrns = item[text_col]
        tag = item[tag_col]
        responses[item[tag_col]] = item[res_col]
        for j in range(len(ptrns)):
            tags.append(tag)
            inputs.append(ptrns[j])

    return tags, inputs, responses


