BÁO CÁO DỰ ÁN AI: XÂY DỰNG ENGINE CONNECT FOUR THÔNG MINH

1. Giới thiệu

Dự án này nhằm mục đích xây dựng một trí tuệ nhân tạo (AI) có khả năng chơi trò chơi "Connect Four" (Cờ Caro 4) một cách hiệu quả và thông minh. Connect Four là một trò chơi chiến thuật hai người chơi đơn giản nhưng đòi hỏi khả năng nhìn trước và đánh giá các nước đi tiềm năng. Mục tiêu của nhóm em là phát triển một engine AI có thể phân tích trạng thái bàn cờ, đánh giá độ mạnh của các vị trí, và chọn nước đi tối ưu nhất trong một khoảng thời gian giới hạn, thể hiện sự tinh tế trong lối chơi.

2. Bài toán và Phương pháp Tiếp cận

Bài toán cốt lõi là tìm nước đi tốt nhất trên bàn cờ Connect Four hiện tại cho một người chơi cụ thể. Do Connect Four là một trò chơi có thông tin hoàn chỉnh (người chơi biết mọi thứ về trạng thái game) và có số lượng trạng thái tương đối hữu hạn so với các game phức tạp hơn như cờ vua, các thuật toán tìm kiếm trên cây trò chơi là phương pháp tiếp cận phù hợp.

Nhóm em lựa chọn sử dụng thuật toán tìm kiếm cây trò chơi Minimax làm nền tảng. Để làm cho Minimax khả thi về mặt tính toán trong thời gian thực, nhóm em kết hợp nó với kỹ thuật tối ưu hóa thiết yếu là Cắt tỉa Alpha-Beta (Alpha-Beta Pruning). Cách tiếp cận này cho phép AI khám phá các kịch bản chơi trong tương lai và đánh giá kết quả tiềm năng của các nước đi.

3. Các Thuật toán và Kỹ thuật Tối ưu hóa Chính

Để nâng cao hiệu quả và sức mạnh của AI, nhóm em đã tích hợp và tinh chỉnh nhiều kỹ thuật tiên tiến. Dưới đây là giải thích chi tiết về cách mỗi thành phần hoạt động và đóng góp vào hiệu suất của engine:

3.1. Thuật toán Minimax (Minimax Algorithm)

Mục đích: Là thuật toán tìm kiếm nền tảng để xác định nước đi tối ưu nhất, giả định cả hai người chơi đều đưa ra quyết định tốt nhất có thể cho bản thân họ. Nó dựa trên nguyên tắc "tối đa hóa lợi ích tối thiểu có thể nhận được" hoặc "tối thiểu hóa thiệt hại tối đa có thể xảy ra".
Cách hoạt động:
Minimax xây dựng một cây trò chơi biểu diễn các trạng thái bàn cờ và các nước đi có thể từ trạng thái hiện tại. Nút gốc là trạng thái bàn cờ hiện tại. Các cạnh là các nước đi hợp lệ, và các nút con là trạng thái bàn cờ sau nước đi đó.
Quá trình tìm kiếm diễn ra theo chiều sâu, khám phá các nhánh cây luân phiên giữa lượt đi của người chơi AI (cố gắng tối đa hóa điểm số) và lượt đi của đối thủ (cố gắng tối thiểu hóa điểm số của AI).
Tìm kiếm dừng lại khi đạt đến một độ sâu tìm kiếm tối đa hoặc khi gặp một trạng thái kết thúc trò chơi (thắng, thua, hòa).
Tại các "nút lá" (các trạng thái cuối cùng của quá trình tìm kiếm), một giá trị số được gán bằng cách sử dụng Hàm Lượng Giá (sẽ giải thích sau). Trạng thái thắng được gán điểm rất cao (+vô cực), trạng thái thua sẽ có điểm rất thấp (-vô cực), hòa có điểm 0.
Sau đó, thuật toán "truyền ngược" (backpropagate) các giá trị từ nút lá lên phía gốc:
Ở các nút của lượt AI (maximizing player), giá trị của nút cha là giá trị lớn nhất trong số các nút con.
Ở các nút của lượt đối thủ (minimizing player), giá trị của nút cha là giá trị nhỏ nhất trong số các nút con.
Giá trị cuối cùng được tính toán cho nút gốc. Nước đi ban đầu dẫn đến nút con có giá trị tương ứng (tại tầng đầu tiên của cây) chính là nước đi tối ưu được Minimax lựa chọn.
Lợi ích: Cung cấp một khuôn khổ logic để đánh giá các nước đi bằng cách xem xét kết quả tiềm năng sau nhiều bước. Đảm bảo AI không mắc những sai lầm cơ bản khi đối thủ chơi tối ưu (trong phạm vi độ sâu tìm kiếm).
3.2. Cắt tỉa Alpha-Beta (Alpha-Beta Pruning)

Mục đích: Là kỹ thuật tối ưu hóa thiết yếu cho thuật toán Minimax, giúp giảm thiểu đáng kể số lượng nút trong cây trò chơi cần phải khám phá mà không làm thay đổi kết quả cuối cùng. Nó hoạt động bằng cách "cắt tỉa" (prune) các nhánh của cây mà thuật toán xác định là không thể dẫn đến kết quả tốt hơn (cho người chơi đang xét) so với những gì đã tìm thấy ở các nhánh khác.
Cách hoạt động:
Thuật toán duy trì hai giá trị: α (alpha) và β (beta) trong quá trình tìm kiếm.
α: Là giá trị tối thiểu mà người chơi Tối đa hóa (AI) có thể đảm bảo nhận được tại thời điểm hiện tại trên nhánh tìm kiếm đang xét.
β: Là giá trị tối đa mà người chơi Tối thiểu hóa (đối thủ) có thể đảm bảo nhận được (hoặc hạn chế AI nhận được) tại thời điểm hiện tại trên nhánh tìm kiếm đang xét.
Ban đầu, α được đặt là −∞ và β là +∞. Các giá trị này được truyền xuống các nút con.
Trong quá trình tìm kiếm:
Khi ở một nút Tối đa hóa (lượt AI): Nếu giá trị tạm thời của một nút con (v) tìm được lớn hơn hoặc bằng giá trị β của nút cha (v >= beta), nhánh tìm kiếm này sẽ bị cắt tỉa. Lý do: người chơi Tối thiểu hóa ở nút cha đã biết rằng họ có thể giới hạn điểm số của AI xuống tối đa β bằng cách chọn một nhánh khác rồi. Vì vậy, nhánh hiện tại dù có cho điểm cao hơn β thì người chơi Tối thiểu hóa cũng sẽ không bao giờ chọn, nên không cần khám phá thêm.
Khi ở một nút Tối thiểu hóa (lượt đối thủ): Nếu giá trị tạm thời của một nút con (v) tìm được nhỏ hơn hoặc bằng giá trị α của nút cha (v <= alpha), nhánh tìm kiếm này sẽ bị cắt tỉa. Lý do: người chơi Tối đa hóa ở nút cha đã biết rằng họ có thể đảm bảo đạt được ít nhất α bằng cách chọn một nhánh khác rồi. Vì vậy, nhánh hiện tại dù có cho điểm thấp hơn α thì người chơi Tối đa hóa cũng sẽ không bao giờ chọn, nên không cần khám phá thêm.
Giá trị α được cập nhật tại các nút Tối đa hóa (α=max(α,giá trị nút con)).
Giá trị β được cập nhật tại các nút Tối thiểu hóa (β=min(β,giá trị nút con)).
Lợi ích: Giảm thiểu đáng kể số lượng trạng thái cần đánh giá, cho phép AI tìm kiếm sâu hơn trong cùng một khoảng thời gian, từ đó đưa ra các nước đi có chiến thuật phức tạp hơn. Hiệu quả tối đa khi các nước đi tốt được xem xét sớm trong quá trình tìm kiếm.
3.3. Hàm Lượng Giá (Evaluation Function)

Mục đích: Cung cấp một cách nhanh chóng để ước lượng "độ tốt" của một trạng thái bàn cờ cho người chơi hiện tại. Nó được sử dụng tại các nút lá không phải trạng thái kết thúc của cây tìm kiếm Minimax/Alpha-Beta để gán giá trị số, hướng dẫn thuật toán đưa ra quyết định.
Cách hoạt động:
Hàm evaluate_board phân tích bàn cờ bằng cách kiểm tra tất cả các chuỗi 4 ô liên tiếp (cửa sổ 4) theo chiều ngang, dọc và cả hai đường chéo.
Đối với mỗi cửa sổ 4 ô (evaluate_window), nó đếm số lượng quân của AI, số lượng quân của đối thủ và số lượng ô trống.
Dựa trên các số lượng này, nó gán điểm số heuristically:
Ví dụ: Cửa sổ có 4 quân của AI được điểm rất cao (chiến thắng).
Cửa sổ có 3 quân AI và 1 ô trống được điểm cao (tạo mối đe dọa hoặc cơ hội tấn công).
Cửa sổ có 2 quân AI và 2 ô trống được điểm thấp hơn.
Các cấu hình tương tự của đối thủ sẽ nhận điểm âm.
AI cũng thưởng điểm cho các quân cờ nằm ở cột trung tâm vì vị trí này tham gia vào nhiều đường thẳng hàng tiềm năng nhất.
Một tối ưu hóa nhỏ là phát hiện "mối đe dọa kép" (hai hoặc nhiều cấu hình 3 quân + 1 trống cùng lúc) và thêm điểm thưởng lớn, khuyến khích AI tạo ra các thế cờ nguy hiểm.
Tổng điểm từ tất cả các cửa sổ và yếu tố khác tạo nên điểm lượng giá cuối cùng, phản ánh ước tính về lợi thế của người chơi AI trong trạng thái bàn cờ đó.
Lợi ích: Cho phép AI đánh giá các trạng thái trung gian của game mà không cần tìm kiếm đến cùng, hướng dẫn Minimax/Alpha-Beta chọn các nhánh có triển vọng tốt, ngay cả khi độ sâu tìm kiếm bị giới hạn.
3.4. Tìm kiếm theo Chiều sâu Lặp (Iterative Deepening - ID)

Mục đích: Quản lý thời gian tìm kiếm một cách hiệu quả và tận dụng thông tin từ các lần tìm kiếm nông hơn. Thay vì chỉ chạy Minimax/Alpha-Beta một lần đến độ sâu tối đa cố định, ID thực hiện nhiều lần tìm kiếm với độ sâu tăng dần.
Cách hoạt động:
Thuật toán bắt đầu bằng việc thực hiện tìm kiếm Alpha-Beta với độ sâu giới hạn là 1.
Nếu vẫn còn thời gian tìm kiếm cho phép, nó sẽ lặp lại quá trình tìm kiếm Alpha-Beta từ đầu với độ sâu giới hạn là 2.
Quá trình này tiếp tục, tăng độ sâu giới hạn lên 3, 4, 5,... cho đến khi thời gian cho phép cho nước đi hiện tại hết (BASE_TIMEOUT được sử dụng để kiểm tra).
Sau mỗi lần tìm kiếm ở một độ sâu d hoàn thành trong thời gian cho phép, nước đi tốt nhất được tìm thấy ở độ sâu đó sẽ được lưu lại như là "nước đi tốt nhất hiện tại".
Nếu quá trình tìm kiếm ở độ sâu d+1 bắt đầu nhưng bị gián đoạn do hết thời gian, AI sẽ trả về nước đi tốt nhất đã được lưu lại từ lần tìm kiếm ở độ sâu d (lần hoàn thành gần nhất).
Lợi ích:
Đảm bảo kết quả trong mọi tình huống: Luôn có một nước đi để trả về khi hết thời gian, đó là nước đi tốt nhất được tìm thấy từ lần lặp hoàn thành cuối cùng.
Phản hồi nhanh: Ở các độ sâu nông, AI nhanh chóng tìm ra các nước đi "khá tốt".
Tận dụng tối ưu hóa: Thông tin về nước đi tốt nhất từ độ sâu d là một gợi ý tuyệt vời để sắp xếp nước đi ở độ sâu d+1, giúp tăng hiệu quả của Alpha-Beta pruning.
3.5. Bảng Chuyển vị (Transposition Table - TT) và Zobrist Hashing

Mục đích: Lưu trữ kết quả của các trạng thái bàn cờ đã được ghé thăm và tính toán trong quá trình tìm kiếm. Điều này tránh việc tính toán lặp lại cho cùng một trạng thái bàn cờ có thể đạt được thông qua các chuỗi nước đi khác nhau (transpositions), giúp tăng tốc độ tìm kiếm đáng kể.
Cách hoạt động (Zobrist Hashing):
Đây là một kỹ thuật hiệu quả để tạo ra một "khóa" số (hash) đại diện cho trạng thái bàn cờ.
Khi khởi tạo, một bảng lớn chứa các số ngẫu nhiên 64-bit duy nhất được tạo ra (ZOBRIST_KEYS). Mỗi số tương ứng với một sự kiện cụ thể: một ô trên bàn cờ có một quân cờ nhất định (quân 1, quân 2) hoặc trống.
Hash của một bàn cờ được tính bằng cách thực hiện phép XOR (bitwise exclusive OR) các số ngẫu nhiên tương ứng với trạng thái của mỗi ô trên bàn cờ đó.
Khi thực hiện một nước đi (thay đổi trạng thái của một ô), hash của bàn cờ mới có thể được cập nhật rất nhanh chóng bằng cách XOR hash cũ với số Zobrist của trạng thái cũ tại ô đó và số Zobrist của trạng thái mới tại ô đó. Việc này hiệu quả hơn nhiều so với việc tính toán lại toàn bộ hash từ đầu.
Cách hoạt động (Transposition Table):
TT là một bộ nhớ đệm (cache) kiểu bảng băm (LimitedDict trong mã nguồn) sử dụng Zobrist hash làm khóa. Mỗi mục trong bảng lưu trữ thông tin về một trạng thái bàn cờ đã được phân tích, bao gồm độ sâu tìm kiếm mà thông tin này được tính, điểm số tìm được, nước đi tốt nhất từ trạng thái đó, và một "cờ" (flag) chỉ ra loại thông tin (exact value, lower bound, upper bound).
Trước khi Minimax/Alpha-Beta bắt đầu phân tích một nút (trạng thái bàn cờ), nó tính Zobrist hash của nút đó và kiểm tra trong TT.
Nếu hash được tìm thấy và thông tin lưu trữ đủ độ sâu (nghĩa là được tính từ một lần tìm kiếm sâu bằng hoặc hơn độ sâu hiện tại) hoặc hữu ích cho việc cắt tỉa (phù hợp với khoảng Alpha-Beta hiện tại), thuật toán sẽ sử dụng thông tin đã lưu và bỏ qua việc tìm kiếm nhánh đó.
Sau khi hoàn thành việc tìm kiếm cho một nút, kết quả sẽ được lưu vào TT.
LimitedDict đảm bảo TT không vượt quá kích thước tối đa bằng cách tự động loại bỏ các mục ít giá trị hơn (thường là các mục từ tìm kiếm nông hơn hoặc ít được truy cập gần đây nhất) khi cần chỗ.
Lợi ích: Tăng đáng kể tốc độ tìm kiếm, cho phép AI khám phá cây trò chơi sâu hơn và rộng hơn trong cùng thời gian, đặc biệt hiệu quả ở các game có nhiều đường đi khác nhau dẫn đến cùng một trạng thái.
3.6. Sắp xếp Nước đi (Move Ordering)

Mục đích: Để tối đa hóa hiệu quả của Cắt tỉa Alpha-Beta. Bằng cách xem xét các nước đi tốt nhất có khả năng trước, thuật toán có cơ hội cao hơn để gặp các cắt tỉa sớm, từ đó giảm số lượng nút cần khám phá.
Cách hoạt động: Hàm sort_moves được sử dụng để sắp xếp danh sách các nước đi hợp lệ trước khi Minimax/Alpha-Beta bắt đầu duyệt qua chúng. Thứ tự ưu tiên được xác định bằng cách kết hợp nhiều heuristic:
Thắng/Cản trực tiếp: Các nước đi dẫn đến chiến thắng ngay lập tức hoặc ngăn chặn đối thủ thắng ngay lập tức được ưu tiên cao nhất.
Đánh giá tĩnh: Sử dụng Hàm Lượng giá để đánh giá tạm thời trạng thái sau khi thực hiện nước đi, cho điểm cao hơn cho các nước đi tạo ra thế trận tốt.
Vị trí trung tâm: Các nước đi vào các cột trung tâm hoặc gần trung tâm được ưu tiên vì chúng tham gia vào nhiều đường chiến thắng hơn.
Killer Moves: Lưu trữ các nước đi đã gây ra cắt tỉa Beta ở các nút "anh em" (sibling nodes) ở cùng độ sâu. Các nước đi này được ưu tiên vì có khả năng cao cũng tốt ở đây.
History Scores: Duy trì một "lịch sử" điểm số cho mỗi cặp (độ sâu, cột). Điểm số này tăng lên mỗi khi nước đi tại cột đó ở độ sâu đó dẫn đến cắt tỉa Beta. Các nước đi có điểm lịch sử cao hơn được ưu tiên.
Danh sách các nước đi được sắp xếp giảm dần dựa trên tổng điểm ưu tiên này theo thứ tự giảm dần.
Lợi ích: Cải thiện đáng kể hiệu suất của Alpha-Beta bằng cách giúp nó "đoán" được các nhánh xấu sớm hơn và cắt tỉa chúng, cho phép AI tìm kiếm sâu hơn hoặc hoàn thành nhanh hơn.
3.7. Cắt tỉa Null Move (Null Move Pruning)

Mục đích: Một heuristic tối ưu hóa bổ sung, chủ yếu được áp dụng ở các nút của người chơi Tối thiểu hóa (đối thủ). Nó dựa trên ý tưởng rằng nếu một vị trí rất mạnh cho người chơi Tối đa hóa ngay cả khi người chơi Tối thiểu hóa được "tặng" một lượt đi miễn phí (null move), thì vị trí đó có khả năng cao là thực sự rất mạnh và có thể gây ra cắt tỉa.
Cách hoạt động: Tại một nút của người chơi Tối thiểu hóa, trước khi xem xét các nước đi thực tế, thuật toán thực hiện một tìm kiếm Alpha-Beta nông hơn (ví dụ: giảm độ sâu đi 3) từ chính trạng thái đó nhưng chuyển lượt ngay sang cho người chơi Tối đa hóa. Nếu kết quả của tìm kiếm nông này cho thấy điểm số cực kỳ tốt cho người chơi Tối đa hóa (đủ để vượt qua ngưỡng Beta hiện tại), thì nhánh của nút Tối thiểu hóa hiện tại có thể bị cắt tỉa. Giả định là nếu AI có lợi thế lớn như vậy ngay cả khi đối thủ "bỏ qua" lượt, thì việc đối thủ đi một nước thực tế cũng khó lòng cải thiện tình hình đủ để tránh bị cắt tỉa.
Lợi ích: Có khả năng tạo ra các cắt tỉa mạnh, đặc biệt ở các vị trí có độ phức tạp cao.
Lưu ý: Đây là một heuristic và có thể không chính xác trong mọi trường huống, đôi khi dẫn đến việc cắt tỉa sai. Tuy nhiên, khi được triển khai và điều chỉnh đúng cách, nó có thể mang lại hiệu quả tăng tốc đáng kể.
4. Chi tiết Triển khai

Engine được triển khai bằng Python, tận dụng thư viện NumPy cho việc thao tác bàn cờ hiệu quả. Các cấu trúc dữ liệu Python chuẩn như dictionary (OrderedDict tùy chỉnh cho Transposition Table) và list được sử dụng để quản lý dữ liệu trò chơi và kết quả tìm kiếm. Việc sử dụng các typing hints giúp mã nguồn rõ ràng và dễ bảo trì hơn.

5. Kết quả và Hiệu suất

Với sự kết hợp nhuần nhuyễn của các thuật toán và kỹ thuật tối ưu hóa đã mô tả, engine AI Connect Four này thể hiện hiệu suất mạnh mẽ. Nó có khả năng tìm kiếm sâu vào cây trò chơi trong thời gian cho phép, phân tích các tình huống phức tạp và đưa ra các nước đi chiến thuật sắc bén. Hàm lượng giá cùng với các heuristic sắp xếp nước đi giúp hướng dẫn tìm kiếm đến các nhánh có triển vọng cao, trong khi Alpha-Beta pruning và Transposition Table loại bỏ phần lớn công việc tính toán lặp thừa. Cơ chế Iterative Deepening và Timeout đảm bảo AI luôn phản hồi kịp thời, ngay cả dưới áp lực thời gian. Engine có thể phát hiện và thực hiện/cản phá các nước đi thắng một cách hiệu quả.

6. Hạn chế và Hướng Phát triển Tiếp theo

Mặc dù đã đạt được hiệu suất tốt, vẫn còn những lĩnh vực mà nhóm em có thể cải tiến:

Độ tinh tế của Hàm Lượng giá: Có thể xem xét các cấu hình quân cờ phức tạp hơn hoặc gán trọng số động cho các yếu tố dựa trên giai đoạn của game.
Điều chỉnh Tham số: Tối ưu hóa các hằng số như độ sâu tìm kiếm tối đa, thời gian chờ, kích thước Transposition Table, và các trọng số trong hàm lượng giá thông qua thử nghiệm rộng rãi hoặc sử dụng kỹ thuật metaheuristics.
Thuật toán Tìm kiếm Nâng cao: Nghiên cứu và triển khai các thuật toán tìm kiếm trạng thái nâng cao hơn như NegaScout, Principal Variation Search (PVS), hoặc MTD(f) có thể mang lại hiệu quả tìm kiếm cao hơn nữa.
Xử lý Cuối Game: Đối với các trạng thái game khi bàn cờ gần đầy, có thể sử dụng một giải thuật chuyên biệt để giải quyết chắc chắn game (thắng, thua, hòa) thay vì chỉ dựa vào lượng giá heuristic.
Cải thiện Heuristics: Tinh chỉnh cách sử dụng Killer Moves và History Scores, hoặc thêm các heuristic sắp xếp nước đi khác.
7. Kết luận

Dự án đã thành công trong việc xây dựng một engine AI chơi Connect Four mạnh mẽ và hiệu quả. Bằng cách áp dụng và kết hợp các thuật toán tìm kiếm cây trò chơi tiên tiến (Minimax, Alpha-Beta Pruning) cùng với các kỹ thuật tối ưu hóa quan trọng (Iterative Deepening, Transposition Table với Zobrist Hashing, Hàm Lượng giá, Sắp xếp nước đi, và Cắt tỉa Null Move), AI có khả năng phân tích game ở độ sâu đáng kể và đưa ra các quyết định chiến thuật thông minh trong thời gian giới hạn. Dự án này cung cấp một nền tảng vững chắc và là minh chứng cho việc ứng dụng hiệu quả các nguyên lý cơ bản của AI vào bài toán giải game đối kháng.

****
