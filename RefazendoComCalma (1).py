import math
import cv2
import cv2.aruco as aruco
import numpy as np

id_to_find = 5  # o id do marcador
marer_size = 10  # o tamanho do marcador em centimetros

# Verificar se a matrix é valida (internet)
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculo de rotação da matrix pelo numeros de euler (internet) // Resultado é dado na mesma ordem do MATLAB
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calibração
calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'intrinsic_matrix (1).txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'distortion_matrix (1).txt', delimiter=',')

# rotacao da matrix em 180 no eixo x (Devido o y da camera ficar invertido em relaçao ao y do marcador aruco
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# definindo o dicionario
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

# capturando a imagem da camera
cap = cv2.VideoCapture(0)

# Difinindo o tamanho da camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

# Fonte para o texto (impresso na imagem)
font = cv2.FONT_HERSHEY_PLAIN

while True:
    # Leitura do frame
    ret, frame = cap.read()

    # convertendo para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # lembrando que no OpenCV as imagens sao salvas em RGB

    # Encontrar todos os marcados aruco da imagem
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None and ids[0] == id_to_find:
        # -- ret = [rvec, tvec, ?]
        # -- array of rotation and position of each marker in camera frame
        # -- rvec = [[rvec_1], [rvec_2], ...]    altitude do marcado pela camera
        # -- tvec = [[tvec_1], [tvec_2], ...]    posição em relação a camera
        ret = aruco.estimatePoseSingleMarkers(corners, marer_size, camera_matrix, camera_distortion)

        # Separando a primeira saida
        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        # Desenhando o marcador
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

        # Printar a posição na tala
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Obtendo a tag de rotação
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc = R_ct.T

        # Pegando as atitude em termos do euler 321 (precisa ser invertido primeiro)
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

        # Printando a altura do marcador referente a camera
        str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            math.degrees(roll_marker), math.degrees(pitch_marker),
            math.degrees(yaw_marker))
        cv2.putText(frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Agr a posição
        pos_camera = -R_tc * np.matrix(tvec).T
        str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Altitude da camera pelo quadro
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
        str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            math.degrees(roll_camera), math.degrees(pitch_camera),
            math.degrees(yaw_camera))
        cv2.putText(frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Frame da tela
    cv2.imshow('frame', frame)

    # fechar o programa apertando "q"
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
