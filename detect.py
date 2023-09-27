import mediapipe as mp
import numpy as np

import pickle

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
model2 = model_dict['model2']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=.1)

def detect(frame):
    x_ = [] 
    y_ = []
    data_aux = []
    try:
        H, W, _ = frame.shape
    
    except Exception as e:
        return f'{e}'
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])#json.loads(r.text)['prediction']
            else:
                prediction = model2.predict([np.asarray(data_aux)])#json.loads(r.text)['prediction']
            return prediction[0]
        except Exception as e:
            return f'{e}'
    return ''

    #     prediction = prediction[0]

        

    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    #     cv2.putText(frame, prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
    #                 cv2.LINE_AA)
    #     withHand += 1
    #     noHand = 0
    #     detected.append(prediction)
    #     if withHand == 5:
    #         words = {x:detected.count(x) for x in detected}
    #         print(words)
    #         word = [k for k,v in words.items() if max(words.values()) == v][0]
    #         wdt.addword(word)
    #         withHand = 0
    #         detected = []
        
    # if noHand == 5:
    #     withHand = 0
    #     detected = []
    #     noHand = 0
    # noHand += 1
    
