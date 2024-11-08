import requests
from bs4 import BeautifulSoup
import arxiv
import openai
from scholarly import scholarly
from PyPDF2 import PdfFileReader
from io import BytesIO
import time


# Set your OpenAI API Key here for summaries (optional)
openai.api_key = ""

def search_papers_arxiv(query, max_results=5):
    """Search for papers on Arxiv using the arxiv API."""
    papers = []
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in search.results():
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published,
                "summary": result.summary,
                "pdf_url": result.pdf_url
            })
    except Exception as e:
        print(f"Error searching Arxiv: {e}")
    return papers

def search_papers_pubmed(query, max_results=5):
    """Search for papers on PubMed using their E-utilities API."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    papers = []
    try:
        search_response = requests.get(search_url, timeout=10).json()
        ids = search_response["esearchresult"]["idlist"]
        for pmid in ids:
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            fetch_response = requests.get(fetch_url, timeout=10)
            soup = BeautifulSoup(fetch_response.content, "xml")
            title = soup.find("ArticleTitle").text if soup.find("ArticleTitle") else "No title"
            abstract = soup.find("AbstractText").text if soup.find("AbstractText") else "No abstract"
            papers.append({"title": title, "summary": abstract, "pdf_url": None})
    except Exception as e:
        print(f"Error searching PubMed: {e}")
    return papers

def search_papers_google_scholar(query, max_results=5):
    """Search for papers on Google Scholar using the scholarly library."""
    papers = []
    try:
        search_query = scholarly.search_pubs(query)
        for i, paper in enumerate(search_query):
            if i >= max_results:
                break
            paper_info = scholarly.fill(paper)
            papers.append({
                "title": paper_info.get("bib", {}).get("title", "No title"),
                "authors": paper_info.get("bib", {}).get("author", "No author"),
                "summary": paper_info.get("bib", {}).get("abstract", "No abstract available"),
                "pdf_url": paper_info.get("eprint_url")  # May be None if unavailable
            })
            time.sleep(1)  # Sleep to prevent rate limiting
    except Exception as e:
        print(f"Error searching Google Scholar: {e}")
    return papers

def download_pdf(url, title):
    """Download PDF if available."""
    if url:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = f"{title.replace(' ', '_')}.pdf"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download: {url}")
        except Exception as e:
            print(f"Error downloading PDF for {title}: {e}")
    else:
        print(f"No PDF URL for: {title}")

def summarize_text(text, max_tokens=150):
    """Summarize text using OpenAI API (optional)."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following text:\n{text}",
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print("Error with OpenAI API:", e)
        return "Summary not available."

def main():
    query = "Sustainability Reporting Europe ESRS CSRD"
    max_results = 5
    
    # Search in Arxiv
    print("Searching in Arxiv...")
    arxiv_papers = search_papers_arxiv(query, max_results)
    
    # Search in PubMed
    print("Searching in PubMed...")
    pubmed_papers = search_papers_pubmed(query, max_results)
    
    # Search in Google Scholar
    print("Searching in Google Scholar...")
    scholar_papers = search_papers_google_scholar(query, max_results)
    
    # Combine all results
    all_papers = arxiv_papers + pubmed_papers + scholar_papers
    
    # Process each paper
    for paper in all_papers:
        print("\nTitle:", paper["title"])
        print("Summary:", paper["summary"])
        
        # Generate a summary if available
        if paper["summary"] != "No abstract available":
            print("Generated Summary:", summarize_text(paper["summary"]))
        
        # Attempt to download PDF if URL is available
        download_pdf(paper["pdf_url"], paper["title"])

if __name__ == "__main__":
    main()