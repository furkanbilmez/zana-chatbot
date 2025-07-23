from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import torch
import random
import re

class KurdishChatbot:
    def __init__(self):
        # Load multilingual sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

        # Predefined question-answer pairs
        self.qa_pairs = [
            ("Merheba", "Slaw, tu çawa yî? Tu ji kû derê?"),
            ("Slaw", "Slaw, heval! Tu çawa yî?"),
            ("Nasılsın?", "Ez baş im, spas dikim! Tu çawa yî?"),
            ("Tu çawa yî?", "Ez baş im, spas! Tu xweşî?"),
            ("Adın ne?", "Navê min Zana ye."),
            ("Sen kimsin?", "Ez chatbotekî te me, Zana."),
            ("Te hez dikim", "Ez jî ji te re hez dikim!"),
            ("Malabadi nerede?", "Malabadi li nêzî Amed e, herî zêde di başûrê Bakurê de ye."),
            ("Amed neresi?", "Amed navê kurdî ya bajarê Diyarbakır e."),
            ("Silvan neresi?", "Silvan bajarê nêzî Amed e."),
            ("Batman neresi?", "Batman bajarê başûrê Bakurê ye."),
            ("Kürtçe öğrenmek istiyorum", "Zor spas! Bi Kurdî biaxivin, ez alîkarî dikim."),
            ("Kürtçe nasıl konuşulur?", "Dest pê bikin bi slaw û roj baş, piştî wê hinek gotinên herî hêsan."),
            ("Türkçe konuşuyorum", "Zor baş! Ji kerema xwe bi Kurdî biaxivî, ez dikarim te alîkarî bikim."),
            ("Başarılar", "Spas heval! Her tim li benda te me."),
            ("İyi günler", "Rojbaş! Bi xêr û silav li ser te."),
            ("Güle güle", "Bi xêr hatî! Roja te xweş be!"),
            ("Rojbaş", "Rojbaş heval! Tu çawa yî?"),
            ("Şev baş", "Şev baş! Xewa te xweş be."),
            ("Yardım edebilir misin?", "Belê, ez ê alîkarî te bikim!"),
            ("Bugün hava nasıl?", "Ez nimûneyek robotim, hêvî dikim tu li derêya xwe rûmet bibî."),
            ("Kaç yaşındasın?", "Ez nexweşî robot im, ez her tim ciwan im!"),
            ("", "Bibore, fêm nakim. Ji kerema xwe ji nû ve bibeje."),
        ]

        self.questions = [q for q, _ in self.qa_pairs]
        self.answers = [a for _, a in self.qa_pairs]
        # Encode all questions once for similarity comparison
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

        # Follow-up questions to keep conversation engaging
        self.follow_up_questions = [
            "Tu çawa têkevî?", 
            "Xwestekî te heye?", 
            "Em dikarin hinek tiştên din qise bikin?", 
            "Tu xwestî tiştên din fêr bibî?", 
            "Te rojbaş çawa bû?",
            "Çi tiştên xweş ji bo min bêje?"
        ]

        # Short automatic replies for quick answers
        self.auto_replies = {
            "başım": "Baş im, spas!",
            "baş": "Zor spas, tu çawa yî?",
            "yaxşı": "Ser çavan!",
            "spas": "Ser çavan, heval!",
            "na": "Ez fêm nakim, ji kerema xwe bikarî zêde bike.",
            "erê": "Baş e, heval.",
            "bextewarim": "Xweş e, hêvî dikim tu jî bextewar bî.",
        }

        # Fun phrases including jokes, proverbs and motivational sentences
        self.fun_phrases = [
            "Zana got: 'Ez robot im, lê qehweya min bê ne ji te re ne!' ☕😂",
            "Heval, ji bo xwe bipirse: 'Çima komputerekê nayê xewn dan?' Bersiva xwe bide!",
            "Zana got: 'Her kes dikare bêrî darê be, lê ne her kes dikare darê be.' 🌳",
            "Zana got: 'Robot me, lê her wisa dilê me heye.' ❤️",
            "Gelo tu dizanî? Jiyan wek stranek e, her kes stranê xwe heye.",
            "Espiriya rojê: 'Dengbêj robot çi got? Beep beep, gotinên te temam ne!' 🤖🎤",
            "Zana dibêje: 'Qehwe bê zêde, jiyan bê reng!' ☕🌈",
            "Zana got: 'Xwedê zimanê wisa daye ku kes fêm nakin, lê hevalan fêm bikin.'",
            "Zana got: 'Tu fêm nakî? Bêhîz be! Ez alîkarî dikim.'",
            "Heval, bêhnedanek ji Zana re: 'Robot me, lê jî dilsoz û keser!' 💙",
        ]

        # Emojis to append to responses for liveliness
        self.emojis = ["🙂", "😊", "😄", "🤖", "☕", "🌟", "🔥", "💬", "🌈", "😂", "💙", "🎉", "😅", "😉", "🤩"]

        # Quick keyword-based answers for common questions
        self.keyword_answers = {
            "nerelisin": "Ez ji dijîtalê me, herî zêde li ser serverên dinyayê.",
            "nasılsın": "Ez baş im, spas dikim! Tu çawa yî?",
            "selam": "Slaw, heval! Tu çawa yî?",
            "merhaba": "Slaw, tu çawa yî? Tu ji kû derê?",
            "teşekkür": "Ser çavan! Her tim amade me ji bo alîkarî.",
            "sevgi": "Ez jî ji te re hez dikim! ❤️",
            
            "çawanî": "Ez baş im, spas dikim! Tu çawa yî?",
            "nasıl gidiyor": "Her tişt baş e, spas! Tu çawa yî?",
            "malabadi": "Malabadi li nêzî Amed e, herî zêde di başûrê Bakurê de ye.",
            "amed": "Amed navê kurdî ya bajarê Diyarbakır e.",
            "silvan": "Silvan bajarê nêzî Amed e.",
            "batman": "Batman bajarê başûrê Bakurê ye.",
        }

        # Special responses when "Furkan" is mentioned
        self.furkan_replies = [
            "Furkan heval, tu her dem li ser xeta rastî yî! ✨",
            "Furkan, gelekî başî, her tiştî ji te re baş e! 🌟",
            "Hey Furkan, tu hêja yî, dest xweş! 👏",
            "Furkan, tu her tim hêja û serkeftî yî! 🚀",
            "Zana dibêje: 'Furkan, hêvîyên te hertim têne bi rastî.' 💙",
            "Furkan, hêvîya min ji bo te her gav zêde dibe! 🔥",
            "Hey Furkan! Tu qehremanê rojê yî! 🦸‍♂️",
        ]

    def normalize_text(self, text):
        # Lowercase and remove punctuation for normalization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def add_emoji(self, text):
        # Append a random emoji for liveliness
        return text + " " + random.choice(self.emojis)

    def get_response(self, user_input):
        user_input_norm = self.normalize_text(user_input)

        # If 'furkan' is mentioned, return a special random reply with emoji
        if "furkan" in user_input_norm:
            return self.add_emoji(random.choice(self.furkan_replies)), False

        # Check keyword-based quick answers first
        for keyword, answer in self.keyword_answers.items():
            if keyword in user_input_norm:
                # 50% chance to add a fun phrase
                if random.random() < 0.5:
                    return self.add_emoji(f"{answer} {random.choice(self.fun_phrases)}"), False
                return self.add_emoji(answer), False

        # Use sentence embeddings to find best matching question
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(user_embedding, self.question_embeddings)[0]
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()

        # Also calculate fuzzy string similarity
        fuzzy_scores = [fuzz.ratio(user_input_norm, q.lower()) for q in self.questions]
        best_fuzzy_idx = fuzzy_scores.index(max(fuzzy_scores))
        best_fuzzy_score = max(fuzzy_scores)

        # Dynamic thresholds based on input length
        length = len(user_input_norm.split())
        if length <= 2:
            cosine_threshold = 0.5
            fuzzy_threshold = 60
        else:
            cosine_threshold = 0.6
            fuzzy_threshold = 70

        # Decide response based on similarity scores and thresholds
        if best_score > cosine_threshold or best_fuzzy_score > fuzzy_threshold:
            answer = self.answers[best_idx] if best_score > cosine_threshold else self.answers[best_fuzzy_idx]

            # 40% chance to add a follow-up question (possibly with fun phrase)
            if random.random() < 0.4:
                follow_up = random.choice(self.follow_up_questions)
                if random.random() < 0.3:
                    follow_up += f" {random.choice(self.fun_phrases)}"
                return self.add_emoji(f"{answer} {follow_up}"), True

            return self.add_emoji(answer), False

        else:
            # If no good match, fallback with a fun phrase and no asking for clarification
            fallback_phrases = [
                "Bibore, fêm nakim. Ji kerema xwe hinek zêdetir bibêje, min alîkarî bikim.",
                "Bibore, fêm nekirim. Tu dikarî hinek din re bêje?",
                "Min hîn naxwazim fêm bikim, dikarî hinek din qise bikî?",
                "Ez fêm nekirim, ji kerema xwe hinek din şirove bike.",
            ]
            fallback = random.choice(fallback_phrases)
            if random.random() < 0.3:
                fallback += f" {random.choice(self.fun_phrases)}"
            return self.add_emoji(fallback), False

    def auto_reply(self, user_input):
        user_input_norm = self.normalize_text(user_input)

        # Special auto reply if 'furkan' mentioned
        if "furkan" in user_input_norm:
            return self.add_emoji(random.choice(self.furkan_replies))

        # Check short auto replies
        for key, val in self.auto_replies.items():
            if key in user_input_norm:
                if random.random() < 0.5:
                    return self.add_emoji(f"{val} {random.choice(self.fun_phrases)}")
                return self.add_emoji(val)
        return None

app = Flask(__name__)
bot = KurdishChatbot()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    auto = data.get("auto", False)

    if not user_message:
        return jsonify({"response": bot.add_emoji("Bibore, agahiyekî binivîse.")})

    if auto:
        reply = bot.auto_reply(user_message)
        if reply:
            return jsonify({"response": reply})
        else:
            return jsonify({"response": ""})

    bot_reply, follow_up_asked = bot.get_response(user_message)
    return jsonify({
        "response": bot_reply,
        "follow_up": follow_up_asked
    })

if __name__ == "__main__":
    app.run(debug=False)
