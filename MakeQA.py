from Agents import GoodStudent, NormalStudent, Teacher, Questioner
from Messages import Question, Answer
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document 
import json
from logging import info
import logging
import os
os.environ["OPENAI_API_KEY"] = "sk-"
from Messages import Sample
from dataclasses import asdict
from tqdm import tqdm


questioner = Questioner()
goodstudent = GoodStudent(student_name="小明")
normalstudent = NormalStudent(student_name="小红")
teacher = Teacher()
students_list = [goodstudent, normalstudent]
output_dir = "/data/lixubin/RetrieveQA/data/output"


def make_QA_pair_from_one_doc(doc:Document) -> list[Sample]:
    dataset = list()
    questions:list[Question] = questioner.invoke(doc)

    show_example_question = '\n\t'.join([questions[i].content[:20]+"..." for i in range(len(questions))])
    print(f"Three new questions! \n\t{show_example_question}\n")

    for index, question in enumerate(questions):
        answers = [goodstudent.invoke(question), normalstudent.invoke(question)]
        preference:Answer = teacher.invoke(answers)
        sample = Sample(**{
                "prompt":question.content, 
                "chosen":preference.content, 
                "reject":[answer for answer in answers if answer.student_name != preference.student_name][0].content,
                "correct":question.correct_answer
            })
        print(f"adding answer for Q{index}: {preference.content[:30]}...")
        dataset.append(sample)
    return dataset

def make_QA_pair(docs:list[Document], save_callback) -> list[Sample]:
    """every single doc will generate three questions."""

    print("Starting...")

    datasets = list()
    
    for doc_index, doc in tqdm(enumerate(docs, start=1)):
        while True:
            retry_times = 1
            if retry_times > 2:
                print(f"""When make QA pairs, error occured over 2 times. Workflow will skip this doc.
                        Error doc index: {doc_index}
                        Error doc content:
                            {doc[:100]}""")
            try:
                datasets += make_QA_pair_from_one_doc(doc)
            except:
                retry_times += 1
                continue
            break
        # here calling save callbacks, set steps here.
        if doc_index > 3:
            save_callback(datasets)
            datasets = list()

    if len(datasets) > 0:
        save_callback(datasets)
        datasets = list()


def to_json_file(dataset:list[Sample], output_dir=output_dir):
    json_dataset = list(map(asdict, dataset))
    file_index = len(os.listdir(output_dir)) + 1
    file_path = os.path.join(output_dir, f"QApair_{file_index}.json")

    with open(file_path, "w") as fp:
        json.dump(json_dataset, fp, ensure_ascii=False, indent=4)

    print(f"Save json to {file_path}.")



if __name__ == "__main__":
    doc1 = Document(page_content="""
社会心理学是心理学的一个主要分支。它所研究的是和社会有关的心理学问题。我们知道，所有的社会事情都有人的因素在里面，也就是都有心理的问题在里面。研究这些课题的心理学就是社会心理学。在当代心理科学中，认知心理学和社会心理学最为人们重视，社会心理学从个体与社会相互作用的角度出发，研究特定社会生活条件下个体心理活动发生发展及其变化的规律。社会心理学不仅强调社会与个体之间的相互作用，重视关于社会情境的探讨，重视个体的内在心理因素。社会心理学的研究范围涉及个体社会心理和社会行为、社会交往心理和行为、群体心理，以及应用社会心理学等层面，即理论与方法、社会个体、态度与行为、社会影响和社会心理学的应用等领域。专业定位是以人文社会科学为导向的社会心理学，并坚持自然科学框架下融合人文社会科学基础的社会心理学研究思路。
回顾西方社会心理学的发展历程，可大致将其划分为几个阶段：
1、西方社会心理学的萌芽与诞生（19世纪末叶到20世纪初）
1908年，美国社会学家罗斯（Ross）的《社会心理学》和英国心理学家麦独孤（McDougall）的《社会心理学导论》 [2]不约而同地发表。西方把这一年作为社会心理学诞生的年代。到了20世纪20年代，美国和苏联的社会心理学家先后把科学实验方法引进这一学科，才使得社会心理学从描述对象转向探索和揭示规律，社会心理学才成为一门独立科学。其奠基人公认为F·奥尔波特。
1924年，美国心理学家奥尔波特（Allport）以实验为基础的《社会心理学》一书的出版，宣告了社会心理学作为一门科学正式开始。 [2]
社会心理学成为一门科学的基础主要有以下三个方面原因：
1）开始运用实验；
2）用数量分析补充对现象的质的分析；
3）从描述现象转向揭示和利用规律。
2、西方社会心理学科学体系的建立（20世纪20年代到40年代）
早在1898年特里普利特关于社会促进的实验研究，可多年来，这个很有价值的提议并没有引起广泛的注意，直到第一次世界大战以后，美国心理学家奥尔波特和德国心理学家默德开创了实验社会心理学方向。在他们之后，实验社会心理学才开始在西方特别是在美国成了社会心理学研究的主流。
1928年瑟斯顿提出了态度测量法，把由托马斯和兹纳涅茨基开始并成为当时社会心理学研究中心的态度研究，提高了一步。1934年莫雷诺提出了社会测量法，用以测量群体内人际吸引和排斥问题。1938年勒温把场论引进社会心理学，这些研究方式集中体现在，依托数学和物理学的原理，为“社会心理学”构建起严谨的科学体系，从而，奠定起它的定量精确研究方向。
3、西方社会心理学的研究领域的扩展
社会心理学研究的主要课题随着时代的演变而有所不同。
《社会心理学》问世
1928年，瑟斯顿提出了态度测量法，把由托马斯和兹纳涅茨基开始，并成为当时社会心理学研究中心的态度研究，提高了一步；1934年，莫雷诺提出了社会测量法，用以测量群体内人际吸引和排斥问题；1938年，勒温把场论引进社会心理学提出了个人生活空间或场的概念，认为行为是个人特点和情境因素相互作用的函数。
研究历史
社会心理学的专题研究，开始于19世纪下半期。1860年出现了拉察鲁斯和斯坦塔尔关于民族心理学的系列论文；此后，塔尔德的《模仿律》、西格尔的《犯罪的群众》、勒庞的《群众心理学》等著作陆续出版，为社会心理学的形成奠定了基础；1908年，英国心理学家麦独孤和美国社会学家罗斯分别出版了社会心理学专著，这标志着社会心理学已成为一门独立的学科。
""")
    doc2 = Document(page_content="""其实
        我个人看的偶像番或者说卖歌番也有一些。Lovelive的几部作品，少女歌剧，佐贺偶像是传奇，偶像荣耀，歌之王子殿下，超时空要塞（虽然不是每个都追到了结尾）……邦邦本家之前虽然没看过，但也听过几首Roselia的曲子。不过说真的，虽然追了好几部，但我个人其实是不算太喜欢偶像番（上述几部里让我觉得真不错可能只有少女歌剧），只是因为这类番剧往往挺轻松，可以听听歌，所以放松时随便一看。偶像番的剧情，通常给我的感觉是不需要太过脑子，人物比较二次元偶像标签化，剧情上每个故事（通常就是每一集）都是为了引出一首歌曲而设计的，最终的目的是卖音乐CD、演唱会门票，固然也有一些煽情的桥段和打动人心的情节，但整体而言设计感往往还是比较强，不像是那些更偏向正剧的番剧能让我全身心地投入剧情里。但MyGo不同，即使不将之看作偶像或者卖歌片，甚至将故事中的歌曲截去、换成一些其他的东西（比如说团队比赛之类的），它的故事也仍然可能成立且很有意思。MyGo的开头对于偶像番是很慢热的，体现为第三集才第一次出现了Live歌曲、而且还使用了偏向于MV的演出（而绝大部分偶像番是第一集就会展示画面精美、歌曲优良的Live，将之作为最大卖点的），前两集除了最开头的乐队解散有点意思，其他部分的节奏都很慢、也没有歌听，说真的如果不是我听别人说第三集很不错，我说不准都坚持不下来追番。而第三集，是本作的神回，也是成功留下我的一集，就像是许多剧情向作品的“三集定律”一样，MyGo厚积薄发，最终在第三集引爆了前两集相对缓慢压抑的气氛与诸多铺垫：第三集的综合素质很高，不仅是剧情上非常出色地塑造了灯、祥子两个人物，解释了老C团如此“白月光”的原因，而且演出也极为精彩——全程都是以灯的第一视角展开的，活用了3D的优势（对于2D来说全程第一人称其实是困难的，但是3D却很容易这样演出），通过灯的第一视角，将她最初的孤独、怯懦，祥子在她心中宛若启示光芒般的地位，她从C团中获得的温暖与希望都非常充分的表达出来。第三集《春日影》这首歌也是点睛之笔，它的词曲就是灯的心声，它仿佛是应这个故事而生的曲子，而非偶像番常见的工业流水线曲（实际上，《春日影》的C团印记很浓厚，C团版本里有键盘的钢琴音色，在我看来其实是更好的一版，不像是作为新MyGo团的主打曲的歌而产出的，因此《春日影》或许就是单纯地为了动画剧情而作的歌），当第三集《春日影》的Live开始的时候，花瓣飘飘、C团的美好记忆纷纷涌来，我甚至没有意识到这是一个乐队/偶像番的Live，而是觉得它是如此合适地在此时出现，去表达灯的情绪，就像是其他非偶像动画里那些抒情的插入曲般。""")
    doc3 = Document(page_content="""故事概要
千早爱音在高中一年级的春天将近结束之际，才由英国回到日本，在奇怪的时间点转学到羽丘女子学园。爱音发现组乐队在学校里蔚为风潮[注 3]，决定也找人来组乐队。她发现在班上被称为“羽丘的怪女生”的高松灯有在写歌词，决定找她加入。殊不知灯早在以前就曾在名为“CRYCHIC”的乐队担任主音和作词，最后不欢而散，以致灯对组乐队十分有阴影。不过，在爱音的鼓励下，灯最终接受了爱音的邀请。CRYCHIC的两名前成员——长崎爽世和椎名立希——也加入。
过程中灯回忆起组成CRYCHIC的经过。灯是个感受性比较特殊的人，从小就觉得自己有点偏离这个世界，跟别人的人际关系不甚顺遂。在中学时，丰川祥子接受了如此奇怪的灯，并邀请她一起组乐队。灯、祥子、爽世、立希和若叶睦组成CRYCHIC。其中一首歌《春日影》由灯作词，祥子作曲，对CRYCHIC来说意义十分重大。她们在第一场演唱会后，祥子却不知道为何退出CRYCHIC，CRYCHIC就此解散。
组成新乐队的四人各有不同的想法。灯还在CRYCHIC解散的阴霾里，希望跟新成员“组一辈子的乐队”。爱音不只不太会弹吉他，还似乎有着其他秘密。爽世很想CRYCHIC重归旧好，得知祥子从月之森转学到羽丘后，便不时到羽丘找祥子谈关于CRYCHIC的事。立希在跟灯在CRYCHIC时，觉得灯把她的心声化为歌声，希望这次组新乐队的灯不再受伤害。
常在Live House“RiNG”流连的要乐奈是个技术很好的吉他手。她觉得灯是个“有趣的女人”，决定加入灯她们的乐队。不久后，因为“RiNG”临时多出一段表演时间，灯她们突然有了举行第一场演唱会的机会。经过一番内心挣扎后，灯下定决心，决定举办演唱会。灯为了这次演唱会写了一首新歌《碧天伴走》，因而她们在演唱会上演唱的不会是《春日影》。
第一场演唱会上，众人都十分紧张，起始表演并不顺畅，灯起初演唱声音还十分小，但看到祥子和睦来看这次演出后才变得大声，众人的演出也开始投入变佳，第一首曲《碧天伴走》演出获得观众的掌声。接着在灯的MC后，乐奈引入了《春日影》，祥子被这一举动震惊，含泪离开现场，并向初华倾诉。演出结束后，虽然其他人十分高兴演出顺利，爽世却生气表示为什么演唱了《春日影》。演出结束后，爽世没再出席乐队的演出，并一直尝试联系祥子道歉，爱音和灯想等爽世来乐队练习，但立希希望不用管爽世；祥子后来约出初华谈心，并透露了她的新计划；爽世在推测出祥子将其拉黑后，通过睦帮忙找到了祥子想当面道歉并解释是为了复活CRYCHIC，但被祥子斥责她只为自己。在被祥子拒绝后，爽世不再理会乐队活动。立希通过姐姐的关系直接找爽世，但爽世说明了就是为了恢复CRYCHIC而重新组建乐队，爱音和乐奈只是意料之外，并知道立希也是这样的想法。立希知道无望后，找了海铃做替补贝斯手，但练习不顺利，海铃不满离开后，立希道出了与爽世交谈的事，知道自己是意料之外后，爱音也退出了练习。
乐队分崩离析，灯认为是自己的错，灯去星象馆时偶然得到初华建议用歌来传达自己的想法；于是灯包下了RiNG全部Live之间的空隙时间来单人唱诵诗歌，乐奈感到有趣也参与了伴奏，甚至也将立希也拉入伴奏中；灯独特的演出引起人们的注意，灯也重新将爱音说服而拉回乐队；之后爱音跟踪爽世回家，并激将她来到演出现场，灯将她强行拉上舞台，众人在即兴演奏的《诗超绊》中相互理解并和好。由于灯预定多了一场三天后的Live时间，于是众人开始准备之后演出的新歌曲和演出服；三天后的Live上，众人演出了《迷星叫》、《迷路的日子》、《碧天伴走》，取得成功。灯根据众人的经历取名乐队为寓意“迷路孩子的乐队”的“MyGO”，爱音在灯乐队命名的基础上加上代表五人的五个感叹号。爽世留意到睦送的黄瓜，追出去将黄瓜退回给睦。
不久后，祥子拉拢到初华和海铃，组建起自己的新乐队，之后也得到了睦的答应，并以此阵容拉拢了若麦的加入。之后祥子的新乐队“Ave Mujica”举行了第一次演出。演出后，祥子自己乘列车回到自己落魄的家中。 """)
    
    make_QA_pair([doc1, doc2, doc3], to_json_file)