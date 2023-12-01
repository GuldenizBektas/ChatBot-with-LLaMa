import arxiv

# papers about LLMs
paper_ids = ['2308.10620', '2307.06435', '2303.18223', '2307.10700', '2310.11207', '2305.11828']

for paper_id in paper_ids:
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    print(paper.title)

    paper.download_pdf("data/")