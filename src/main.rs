use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::rc::Rc;
use std::str;

use gtrie::Trie;

const PATH_NAME: &str = "index";
const TERM_DICT_FILE_NAME: &str = "terms_dict.dat";
const POSTING_LISTS_FILE_NAME: &str = "posting_lists.dat";

struct Segment {
    dict: Trie<char, PostingList>,
    docs: HashMap<i64, Rc<Document>>,
}

#[derive(Clone)]
struct Document {
    id: i64,
    text: String,
}

#[derive(Clone)]
struct PostingNode {
    doc_id: i64,
    freq: i32,
}

#[derive(Clone)]
struct PostingList {
    list: Vec<PostingNode>,
}

#[derive(PartialEq, Copy, Clone)]
struct F32(f32);

impl Eq for F32 {}

impl PartialOrd for F32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for F32 {
    fn cmp(&self, other: &F32) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Copy, Clone, PartialEq)]
struct TopKDoc {
    id: i64,
    score: F32,
}

impl PartialOrd<Self> for TopKDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for TopKDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        return other.score.cmp(&self.score);
    }
}

impl Eq for TopKDoc {}

pub trait Tokenizer {
    fn tokenize(&mut self, text: String) -> Vec<(String, i32)>;
}

struct NaiveTokenizer {}

impl NaiveTokenizer {
    pub fn new() -> Self {
        NaiveTokenizer {}
    }
}

impl Tokenizer for NaiveTokenizer {
    fn tokenize(&mut self, text: String) -> Vec<(String, i32)> {
        let tokens: Vec<String> = text.split_whitespace().map(|v| { v.to_string() }).collect();
        let mut result = Vec::new();
        for x in tokens {
            result.push((x, 0));
        }
        return result;
    }
}

fn main() {
    println!("Hello, world!");
    let _ = init();
}

fn init() -> std::io::Result<()> {
    fs::create_dir_all("index")?;
    let term_dict_file = File::create(format!("{}/{}", PATH_NAME, TERM_DICT_FILE_NAME))?;
    let posting_lists_file = File::create(format!("{}/{}", PATH_NAME, POSTING_LISTS_FILE_NAME))?;
    Ok(())
}

fn index_documents(documents: Vec<Document>) -> std::io::Result<Segment> {
    let mut tokenizer = NaiveTokenizer::new();

    let mut dict: Trie<char, PostingList> = Trie::new();
    let mut docs = HashMap::new();
    for document in documents {
        let doc_id = document.id.clone();
        let doc_text = document.text.clone();
        let link_to_doc = Rc::new(document);

        docs.insert(doc_id, link_to_doc);
        let tokens = &tokenizer.tokenize(doc_text);

        for x in tokens {
            let token = &x.0;

            let posting = dict.get_value(token.chars())
                .unwrap_or(PostingList { list: Vec::new() });

            let mut updated = false;
            let mut updated_posting = posting.list;
            for i in 0..updated_posting.len() {
                if updated_posting[i].doc_id == doc_id {
                    updated_posting[i].freq += 1;
                    updated = true;
                    break;
                }
            }
            if !updated {
                updated_posting.push(PostingNode { doc_id, freq: 1 });
            }
            dict.insert(token.chars(), PostingList { list: updated_posting });
        }
    }

    Ok(Segment { dict, docs })
}

// fn flush_to_disk() {
//     let term_dict_file = File::create(format!("{}/{}", PATH_NAME, TERM_DICT_FILE_NAME))?;
//     let posting_lists_file = File::create(format!("{}/{}", PATH_NAME, POSTING_LISTS_FILE_NAME))?;
//
//     let mut term_bw = BufWriter::new(term_dict_file);
//     let mut posting_bw = BufWriter::new(posting_lists_file);
// }

fn score_tf_idf(term_freq: i32, total_docs_with_term: i32, total_docs_in_segment: i32) -> f32 {
    return if total_docs_with_term == 0 {
        0 as f32
    } else {
        let base: f32 = (total_docs_in_segment as f32 / total_docs_with_term as f32);
        return term_freq as f32 * base.log2();
    };
}

fn search(segment: Segment, query: String) -> Vec<TopKDoc> {
    let terms = segment.dict.get_value(query.chars());
    let mut top_k = BinaryHeap::new();

    return if terms.is_none() {
        vec![]
    } else {
        let total_doc_segment = segment.docs.len();
        let final_terms = terms.unwrap();
        let total_doc_with_term = final_terms.list.len();
        for i in 0..final_terms.list.len() {
            let score = score_tf_idf(final_terms.list[i].freq,
                                     total_doc_with_term as i32,
                                     total_doc_segment as i32);
            top_k.push(TopKDoc { id: final_terms.list[i].doc_id, score: F32(score) });
        }

        let mut result = Vec::new();
        while let Some(doc) = top_k.pop() {
            result.push(doc);
        }
        return result;
    };
}

#[cfg(test)]
mod tests {
    use crate::{Document, index_documents, NaiveTokenizer, search, Tokenizer};

    #[test]
    fn tokenize_success() {
        let mut tokenizer = NaiveTokenizer::new();
        let tokens = tokenizer.tokenize(String::from("hello this is a text"));
        for x in &tokens {
            assert_eq!(x.1, 0);
        }
        assert_eq!(tokens[0].0, "hello");
        assert_eq!(tokens[1].0, "this");
        assert_eq!(tokens[2].0, "is");
        assert_eq!(tokens[3].0, "a");
        assert_eq!(tokens[4].0, "text");
    }

    #[test]
    fn index_success() {
        let doc_1 = Document { id: 1, text: "hello this is test".to_string() };
        let doc_2 = Document { id: 2, text: "hello second test test".to_string() };
        let doc_3 = Document { id: 3, text: "hello".to_string() };
        let docs = vec![doc_1, doc_2, doc_3];
        let segment = index_documents(docs).expect("");
        let posting = segment.dict.get_value("test".chars()).expect("").list;
        assert_eq!(posting[0].freq, 1);
        assert_eq!(posting[0].doc_id, 1);

        assert_eq!(posting[1].freq, 2);
        assert_eq!(posting[1].doc_id, 2);
    }

    #[test]
    fn search_success() {
        let doc_1 = Document { id: 1, text: "hello this is test".to_string() };
        let doc_2 = Document { id: 2, text: "hello second test test".to_string() };
        let doc_3 = Document { id: 3, text: "hello".to_string() };
        let docs = vec![doc_1, doc_2, doc_3];
        let segment = index_documents(docs).expect("");
        let found_docs = search(segment, "test".to_string());
        assert_eq!(found_docs.len(), 2);
        assert_eq!(found_docs[1].id, 1);
        assert_eq!(found_docs[0].id, 2);
    }
}
