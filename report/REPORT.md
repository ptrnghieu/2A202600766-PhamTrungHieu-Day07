# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Trung Hiếu
**Nhóm:** B5
**Ngày:** 5/6/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai đoạn văn bản có vector embedding trỏ cùng hướng trong không gian nhiều chiều, tức là chúng mang ý nghĩa hoặc ngữ cảnh tương tự nhau. Điểm càng gần 1 thì hai đoạn càng gần về mặt ngữ nghĩa, bất kể dùng chính xác cùng từ hay không.

**Ví dụ HIGH similarity:**
- Sentence A: `Người lao động có quyền đơn phương chấm dứt hợp đồng khi không được trả đủ lương.`
- Sentence B: `NLĐ được phép tự ý kết thúc hợp đồng nếu tiền lương không được thanh toán đầy đủ.`
- Tại sao tương đồng: Hai câu diễn đạt cùng một quyền pháp lý bằng từ ngữ khác nhau; embedding của cả hai đều tập trung vào chủ đề chấm dứt hợp đồng và thanh toán lương.

**Ví dụ LOW similarity:**
- Sentence A: `Người lao động có quyền đơn phương chấm dứt hợp đồng khi không được trả đủ lương.`
- Sentence B: `Thời tiết hôm nay nắng đẹp, phù hợp để tổ chức dã ngoại ngoài trời.`
- Tại sao khác: Câu đầu thuộc chủ đề luật lao động, câu thứ hai về thời tiết; hai ngữ cảnh hoàn toàn không liên quan, vector hướng về hai vùng rất xa nhau trong không gian embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector chứ không đo khoảng cách tuyệt đối, nên không bị ảnh hưởng bởi độ dài văn bản. Một câu ngắn và một đoạn dài cùng chủ đề có thể có cosine similarity cao dù độ dài vector rất khác nhau; nếu dùng Euclidean distance thì câu dài sẽ luôn bị coi là "xa hơn" dù nghĩa giống nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks`. Số chunk tăng từ 23 lên 25 vì bước nhảy nhỏ hơn. Overlap lớn giúp giữ ngữ cảnh liên tục ở ranh giới chunk, giảm nguy cơ cắt mất điều kiện/ngoại lệ pháp lý nằm ngay chỗ chuyển tiếp.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Văn bản pháp luật Việt Nam — ba lĩnh vực: Bảo hiểm xã hội, Hôn nhân & Gia đình, Lao động.

**Tại sao nhóm chọn domain này?**
> Pháp luật là loại dữ liệu có cấu trúc rõ ràng (chương, mục, điều, khoản) nhưng câu hỏi người dùng thường bằng ngôn ngữ tự nhiên, đòi hỏi retrieval chính xác chứ không thể suy đoán. Câu trả lời phải có nguồn cụ thể và không được thiếu điều kiện hay ngoại lệ, đây là thử thách lý tưởng để đánh giá chất lượng chunking và RAG. Ba bộ luật được chọn đủ đa dạng về thuật ngữ để kiểm tra khả năng phân biệt domain khi filter metadata.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `luat_bhxh_I.txt` | `data/luat_bhxh_I.txt` | 17,256 | `law=bhxh`, `section=I`, `source=luat_bhxh_I` |
| 2 | `luat_bhxh_IV.txt` | `data/luat_bhxh_IV.txt` | 21,696 | `law=bhxh`, `section=IV`, `source=luat_bhxh_IV` |
| 3 | `luat_bhxh_V.txt` | `data/luat_bhxh_V.txt` | 72,778 | `law=bhxh`, `section=V`, `source=luat_bhxh_V` |
| 4 | `luat_hon_nhan_III.txt` | `data/luat_hon_nhan_III.txt` | 18,360 | `law=hon_nhan`, `section=III`, `source=luat_hon_nhan_III` |
| 5 | `luat_hon_nhan_V.txt` | `data/luat_hon_nhan_V.txt` | 25,573 | `law=hon_nhan`, `section=V`, `source=luat_hon_nhan_V` |
| 6 | `luat_lao_dong_III.txt` | `data/luat_lao_dong_III.txt` | 40,176 | `law=lao_dong`, `section=III`, `source=luat_lao_dong_III` |
| 7 | `luat_lao_dong_V.txt` | `data/luat_lao_dong_V.txt` | 23,271 | `law=lao_dong`, `section=V`, `source=luat_lao_dong_V` |

*(Tổng corpus: 18 file, ~405,000 ký tự)*

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `law` | string | `bhxh`, `hon_nhan`, `lao_dong` | Filter theo bộ luật để tránh lấy chunk từ domain không liên quan khi câu hỏi rõ ngữ cảnh. |
| `section` | string | `I`, `III`, `V` | Xác định chương/phần trong luật, hỗ trợ debug khi chunk sai lại biết ngay nằm ở đâu. |
| `source` | string | `luat_bhxh_V` | Hiển thị nguồn chunk cho người dùng và agent để grounding câu trả lời. |
| `doc_id` | string | `luat_lao_dong_III_c12` | Khóa để delete_document và truy vết chunk trong store khi cần cập nhật corpus. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện với `chunk_size=400`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `luat_bhxh_I.txt` (17,256 c) | FixedSizeChunker (`fixed_size`) | 50 | 394 | Trung bình — kích thước ổn định nhưng thường cắt ngang giữa điều/khoản. |
| `luat_bhxh_I.txt` (17,256 c) | SentenceChunker (`by_sentences`) | 43 | 399 | Trung bình — giữ câu nguyên vẹn nhưng với văn bản luật một "câu" thường rất dài nên chunk ít đều hơn. |
| `luat_bhxh_I.txt` (17,256 c) | RecursiveChunker (`recursive`) | 65 | 264 | Cao hơn — ưu tiên tách theo `\n\n` và `\n` nên giữ được ranh giới điều/khoản. |
| `luat_hon_nhan_I.txt` (10,301 c) | FixedSizeChunker | 30 | 392 | Trung bình |
| `luat_hon_nhan_I.txt` (10,301 c) | SentenceChunker | 31 | 330 | Trung bình |
| `luat_hon_nhan_I.txt` (10,301 c) | RecursiveChunker | 40 | 256 | Cao hơn |
| `luat_lao_dong_I.txt` (10,088 c) | FixedSizeChunker | 29 | 396 | Trung bình |
| `luat_lao_dong_I.txt` (10,088 c) | SentenceChunker | 28 | 358 | Trung bình |
| `luat_lao_dong_I.txt` (10,088 c) | RecursiveChunker | 35 | 286 | Cao hơn |

### Strategy Của Tôi

**Loại:** Tuned `SentenceChunker` và `FixedSizeChunker` — thử nhiều tham số để tìm cấu hình tốt nhất cho corpus luật.

**Mô tả cách hoạt động:**
> `SentenceChunker` tách văn bản tại các ký tự kết thúc câu (`.`, `!`, `?`) theo sau bởi khoảng trắng, sau đó gom `max_sentences_per_chunk` câu thành một chunk. `FixedSizeChunker` cắt mỗi `chunk_size` ký tự với phần `overlap` dùng chung giữa hai chunk liền kề để duy trì ngữ cảnh ranh giới. Cả hai được tuning với nhiều tham số khác nhau trên corpus luật để tìm điểm cân bằng giữa chunk count và avg length.

**Tham số đã thử nghiệm (`luat_bhxh_I.txt` làm mẫu):**

| Variant | Chunk Count | Avg Length |
|---------|-------------|------------|
| SentenceChunker(max=2) | 64 | 268 |
| SentenceChunker(max=3) | 43 | 399 |
| **SentenceChunker(max=4)** | **32** | **536** |
| SentenceChunker(max=5) | 26 | 660 |
| FixedSizeChunker(200, 20) | 96 | 200 |
| FixedSizeChunker(400, 50) | 50 | 394 |
| **FixedSizeChunker(600, 80)** | **34** | **585** |
| FixedSizeChunker(800, 100) | 25 | 786 |

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tôi chọn hai strategy này để làm baseline so sánh cho nhóm, đồng thời tune tham số để thấy rõ giới hạn của từng cách. `SentenceChunker(max=4)` và `FixedSizeChunker(600, 80)` là hai cấu hình tốt nhất sau khi thử, cả hai đều tạo chunk ~500–600 ký tự — đủ để chứa một quy định pháp lý ngắn mà không quá dài để gây nhiễu retrieval. Đây là baseline để các thành viên khác trong nhóm so sánh strategy phức tạp hơn của họ.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `luat_bhxh_I.txt` | best baseline: RecursiveChunker | 65 | 264 | Tốt nhất trong 3 baseline về context preservation nhờ tách theo cấu trúc văn bản. |
| toàn corpus (18 file) | SentenceChunker(max=4) | 610 | ~490 | Tương đương recursive về số lượng; giữ câu nguyên vẹn hơn fixed-size. |
| toàn corpus (18 file) | **FixedSizeChunker(600, 80)** | **587** | ~560 | Ít chunk hơn sentence chunker, overlap 80 giúp giảm mất ngữ cảnh ở ranh giới, nhưng vẫn cắt ngang điều/khoản. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Phạm Trung Hiếu | SentenceChunker + FixedSizeChunker (tuned) | 2 / 10 | Đơn giản, nhanh, dễ tune; phù hợp để làm baseline so sánh | Không hiểu cấu trúc pháp lý, hay cắt nhầm điều kiện/ngoại lệ khỏi điều khoản tương ứng |
| Lưu Thiện Việt Cường | Agentic chunking | 10 / 10 | Chunk theo đơn vị nghĩa pháp lý, title/summary/topics giúp LLM trả lời đủ điều kiện + ngoại lệ | Tốn API, phụ thuộc chất lượng JSON của LLM |
| Nguyễn Anh Quân | Parent-child chunking | 10 / 10 | Cân bằng child chunk nhỏ để retrieve chính xác và parent context để giữ bối cảnh rộng | Cần quản lý mapping parent-child |
| Đinh Nhật Thành | Late chunking | 8.7 / 10 | Giữ ngữ cảnh tài liệu trước khi embedding, tốt hơn rõ rệt so với chunking thường | Cần model/API hỗ trợ late chunking |
| Nguyễn Khánh Toàn | StructureChunker | 3.3 / 10 | Tận dụng heading pháp lý (Chương/Mục/Điều), giữ ranh giới văn bản tốt hơn fixed-size | Fail nhiều câu hỏi đòi chi tiết sâu trong văn bản |
| Phạm Thị Linh Chi | LawChunker | 3.3 / 10 | Khai thác cấu trúc Điều/Khoản, tránh mất ranh giới pháp lý | Chưa ổn định trên toàn bộ 15 câu benchmark |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Agentic chunking và parent-child là hai hướng hiệu quả nhất cho domain pháp luật. Domain này đặc biệt khó vì câu trả lời cho một câu hỏi thường cần đủ điều kiện chính + ngoại lệ + phạm vi áp dụng — tất cả có thể nằm ở các điều/khoản khác nhau. Strategy hiểu cấu trúc pháp lý (agentic) hoặc giữ cả context rộng lẫn chunk nhỏ để retrieve (parent-child) xử lý được vấn đề này, trong khi chunking theo số ký tự hay câu văn không có cơ chế đó.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[.!?])\s+', text.strip())` để tách text tại dấu kết thúc câu theo sau bởi khoảng trắng; loại phần rỗng sau split rồi gom tối đa `max_sentences_per_chunk` câu vào một chunk bằng vòng lặp bước `max_sentences_per_chunk`. Edge case: nếu regex không tìm thấy ranh giới câu (ví dụ văn bản không có dấu chấm), toàn bộ text được trả về như một chunk duy nhất.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk` gọi `_split(text, self.separators)`. `_split` kiểm tra base case (text ≤ chunk_size) trước; nếu separator không tìm thấy trong text thì tiếp tục với separator tiếp theo; nếu separator là chuỗi rỗng thì cắt cứng theo `chunk_size`. Sau khi split, các phần nhỏ được gom lại bằng cách nối với separator cho đến khi đạt `chunk_size`; phần nào vượt quá thì đệ quy với `next_separators`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `_make_record` embed nội dung document rồi đóng gói thành dict `{id, content, embedding, metadata}` với `doc_id` được inject thêm vào metadata. `add_documents` gọi `_make_record` cho từng doc và append vào `_store` (in-memory). `search` embed query rồi gọi `_search_records` tính dot product giữa query vector và mọi stored embedding, sort giảm dần và trả về top_k dict có key `content`, `metadata`, `score`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` với `metadata_filter=None` delegate thẳng sang `search`; khi có filter thì list comprehension lọc `_store` bằng equality trên từng key-value trước, rồi `_search_records` chỉ chạy trên tập đã lọc. `delete_document` xây list mới loại bỏ mọi record có `metadata["doc_id"] == doc_id`, trả `True` nếu length giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve `top_k` chunks từ store, nối nội dung bằng `"\n\n".join(...)` thành phần context, ghép vào prompt dạng `"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"` rồi call `llm_fn`. Prompt đặt context trước question để LLM bám vào nguồn tài liệu trước khi trả lời.

### Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.0.2

collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Chạy với `_mock_embed` (MockEmbedder, dim=64) — vector ngẫu nhiên theo hash nên tất cả scores gần 0.*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Lao động nữ sinh con phải đóng BHXH từ đủ 06 tháng trong 12 tháng trước khi sinh. | Điều kiện hưởng chế độ thai sản yêu cầu đóng bảo hiểm ít nhất 6 tháng trong một năm liền kề trước sinh. | high | 0.1125 | Sai — cùng nghĩa nhưng score thấp |
| 2 | Mức lương hưu hằng tháng được tính bằng tỷ lệ hưởng nhân với mức bình quân tiền lương đóng BHXH. | Nhà hàng hải sản ven biển mở cửa từ 9 giờ sáng đến 10 giờ tối mỗi ngày. | low | 0.0563 | Đúng — gần 0, hai chủ đề không liên quan |
| 3 | Người sử dụng lao động phải báo trước ít nhất 45 ngày khi đơn phương chấm dứt hợp đồng không xác định thời hạn. | Người thuê lao động cần thông báo trước tối thiểu 45 ngày khi kết thúc hợp đồng vô thời hạn. | high | 0.1007 | Sai — cùng nghĩa nhưng score thấp |
| 4 | Con dưới 36 tháng tuổi khi ly hôn được giao cho người mẹ trực tiếp nuôi dưỡng. | Trẻ em chưa đủ 3 tuổi thường được ưu tiên giao cho mẹ chăm sóc sau khi cha mẹ ly hôn. | high | 0.0381 | Sai — cùng nghĩa nhưng score rất thấp |
| 5 | Tài sản chung của vợ chồng gồm thu nhập do lao động trong thời kỳ hôn nhân và tài sản tặng cho chung. | Người lao động được đơn phương chấm dứt hợp đồng không cần báo trước khi không được trả đủ lương. | low | 0.1415 | Sai — score cao nhất dù hai câu khác domain |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là pair 1, 3, 4: hai câu gần như cùng nghĩa nhưng score chỉ đạt 0.10–0.11, còn pair 5 — hai câu khác domain hoàn toàn — lại có score cao nhất (0.14). Điều này phơi bày giới hạn căn bản của `_mock_embed`: nó tạo vector ngẫu nhiên theo hash MD5 của chuỗi ký tự, không có bất kỳ hiểu biết về ngữ nghĩa. Kết quả nhắc rõ rằng chất lượng embedding model là yếu tố quyết định — chunking strategy dù tốt đến đâu cũng vô nghĩa nếu embedding không nắm được nghĩa văn bản.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 trong 15 benchmark queries của nhóm trên `SentenceChunker(max_sentences_per_chunk=4)` với `_mock_embed`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Điều kiện thời gian đóng BHXH để lao động nữ hưởng chế độ thai sản khi sinh con? | Đóng BHXH bắt buộc đủ 06 tháng trong 12 tháng trước sinh; trường hợp dưỡng thai cần đủ 03 tháng nếu đã đóng đủ 12 tháng. |
| 2 | Công dân Việt Nam cần điều kiện gì để hưởng trợ cấp hưu trí xã hội? | Từ đủ 75 tuổi, không hưởng lương hưu/trợ cấp BHXH hằng tháng, có văn bản đề nghị; ngoại lệ cho nhóm 70–75 tuổi nghèo/cận nghèo. |
| 3 | Nam và nữ cần đáp ứng điều kiện tuổi gì để kết hôn hợp pháp? | Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi. |
| 4 | Thời gian thử việc tối đa với công việc cần trình độ từ cao đẳng trở lên? | Không quá 60 ngày. |
| 5 | Thời hạn báo trước khi NSDLĐ đơn phương chấm dứt HĐLĐ không xác định thời hạn? | Ít nhất 45 ngày; trừ một số trường hợp đặc biệt theo luật. |

### Kết Quả Của Tôi

**Cấu hình:** `SentenceChunker(max_sentences_per_chunk=4)`, `_mock_embed`, in-memory store, tổng 610 chunks.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer |
|---|-------|--------------------------------|-------|-----------|--------------|
| 1 | Điều kiện thời gian đóng BHXH cho thai sản? | Trách nhiệm cơ quan BHXH đôn đốc thu (*luat_bhxh_IV*) | 0.3606 | Không — đúng domain nhưng sai nội dung | N/A (mock LLM) |
| 2 | Điều kiện hưởng trợ cấp hưu trí xã hội? | Cho thuê lại lao động có điều kiện (*luat_lao_dong_III*) | 0.3338 | Không — sai domain | N/A |
| 3 | Điều kiện tuổi kết hôn? | Quyền/nghĩa vụ người nước ngoài trong hôn nhân VN (*luat_hon_nhan_VIII*) | 0.3455 | Một phần — đúng domain, sai điều khoản | N/A |
| 4 | Thời gian thử việc cao đẳng trở lên? | Điều kiện mang thai hộ (*luat_hon_nhan_V*) | 0.3810 | Không — sai domain | N/A |
| 5 | Thời hạn báo trước chấm dứt HĐLĐ? | Điều kiện hưởng BHXH (*luat_bhxh_V*) | 0.3537 | Không — sai domain | N/A |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **0 / 5**

> **Lý do:** `_mock_embed` tạo vector ngẫu nhiên hoàn toàn dựa trên hash chuỗi ký tự, không có semantic signal. Scores nằm trong khoảng 0.31–0.38 không phân biệt được query liên quan hay không liên quan — kết quả retrieval gần như ngẫu nhiên. Đây là failure case chính của toàn bộ pipeline khi dùng mock embedder trên corpus thực.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Kết quả của nhóm cho thấy rõ sự chênh lệch lớn giữa các strategy: agentic và parent-child đạt 15/15, late chunking đạt 13/15, còn baseline chỉ 2–3/15. Điều tôi học được từ Lưu Thiện Việt Cường là chunking theo đơn vị ý nghĩa pháp lý (thay vì số ký tự hay số câu) thực sự tạo ra khác biệt vì một quy định pháp lý hoàn chỉnh thường gồm điều kiện chính + ngoại lệ + phạm vi áp dụng — nếu chunk cắt nhầm thì LLM sẽ thiếu thông tin dù retrieve đúng file. Từ Đinh Nhật Thành, tôi thấy late chunking là hướng đáng để thử vì nó giữ ngữ cảnh tài liệu mà không cần LLM làm chunker.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Từ demo của các nhóm khác, tôi thấy không chỉ chunking mà metadata schema cũng ảnh hưởng lớn đến chất lượng retrieval: những nhóm gắn metadata chi tiết như `article_number`, `clause` hay `legal_topic` cho phép filter chính xác hơn và giúp agent trace nguồn dễ hơn. Ngoài ra, có nhóm chia sẻ rằng với corpus luật, hybrid search (kết hợp BM25 keyword và embedding vector) cho kết quả ổn định hơn embedding thuần vì các câu hỏi pháp lý thường chứa số điều/khoản cụ thể mà BM25 bắt chính xác còn embedding hay bỏ sót.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Trước tiên tôi sẽ dùng embedding model thực (ít nhất `all-MiniLM-L6-v2`) thay vì mock để retrieval có semantic signal thực sự — đây là bottleneck lớn nhất trong experiment này. Về chunking, tôi sẽ chuyển sang RecursiveChunker với separator ưu tiên `\nĐiều ` và `\n\d+\.` để tận dụng cấu trúc điều/khoản của văn bản luật, đồng thời thêm metadata `article_number` và `clause` để `search_with_filter` có thể filter chính xác hơn theo điều luật cụ thể.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 6 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **81 / 100** |
