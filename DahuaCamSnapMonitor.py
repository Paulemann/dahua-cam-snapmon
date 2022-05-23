#!/usr/bin/env python3

# pip install requests
# pip install nummpy
# pip install opencv-contrib-python
# pip install imutils

import configparser
import argparse
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests.exceptions import RequestException, ConnectionError
from datetime import datetime

import os, sys
import numpy as np
import cv2
from io import BufferedReader
from time import sleep, time
from multiprocessing import Process, Event, active_children
from imutils.object_detection import non_max_suppression

import smtplib
from email.utils import formataddr
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders


# Pretrained classes in the model
classNames = {0: 'Background',
              1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane', 6: 'Bus',
              7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic light', 11: 'Fire Hydrant',
              13: 'Stop Sign', 14: 'Parking Meter', 15: 'Bench', 16: 'Bird', 17: 'Cat',
              18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant', 23: 'Bear',
              24: 'Zebra', 25: 'Giraffe', 27: 'Backpack', 28: 'Umbrella', 31: 'Handbag',
              32: 'Tie', 33: 'Suitcase', 34: 'Frisbee', 35: 'Skis', 36: 'Snowboard',
              37: 'Sports Ball', 38: 'Kite', 39: 'Baseball Bat', 40: 'Baseball Glove',
              41: 'Skateboard', 42: 'Surfboard', 43: 'Tennis Racket', 44: 'Bottle',
              46: 'Wine Glass', 47: 'Cup', 48: 'Fork', 49: 'Knife', 50: 'Spoon',
              51: 'Bowl', 52: 'Banana', 53: 'Apple', 54: 'Sandwich', 55: 'Orange',
              56: 'Broccoli', 57: 'Carrot', 58: 'Hot Dog', 59: 'Pizza', 60: 'Donut',
              61: 'Cake', 62: 'Chair', 63: 'Couch', 64: 'Potted Plant', 65: 'Bed',
              67: 'Dining Table', 70: 'Toilet', 72: 'TV', 73: 'Laptop', 74: 'Mouse',
              75: 'Remote', 76: 'Keyboard', 77: 'Cell Phone', 78: 'Microwave', 79: 'Oven',
              80: 'Toaster', 81: 'Sink', 82: 'Refrigerator', 84: 'Book', 85: 'Clock',
              86: 'Vase', 87: 'Scissors', 88: 'Teddy Bear', 89: 'Hair Drier', 90: 'Toothbrush'}

modelFile   = './models/frozen_inference_graph.pb'
configFile  = './models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


def log(message):
	print(f"{datetime.now().replace(microsecond=0)} {message}")


class Mailer():
	def __init__(self, server='', port=587, use_tls=True, sender='', user='', password=''):
		self.server 	= server
		self.port 		= port
		self.use_tls 	= use_tls
		self.user 		= user
		self.password 	= password
		self.sender		= sender

		self.recipients 	= []
		self.attachments 	= []

		log('[SMTP] Initialized')


	def attach(self, data, name):
		self.attachments.append({'data': data, 'name': name})


	def reset_attachments(self):
		self.attachments = []


	def add_recipient(self, address):
		self.recipients.append(address)


	def send(self, subject, message):
		if not self.recipients:
			log('[SMTP] No recipients')
			return False

		msg = MIMEMultipart()

		if self.sender:
			msg['From'] = formataddr((str(Header(self.sender, 'utf-8')), self.user))
		else:
			msg['From'] = self.user
		msg['To'] = ', '.join(self.recipients)
		msg['Subject'] = subject

		msg.attach(MIMEText(message, 'plain'))

		if self.attachments:
			for attachment in self.attachments:
				part = MIMEBase('application', 'octet-stream')
				part.set_payload(attachment['data'])
				encoders.encode_base64(part)
				part.add_header('Content-Disposition',
		                        'attachment; filename={}'.format(attachment['name']))
				msg.attach(part)

		try:
			smtp = smtplib.SMTP(self.server, self.port)

			if self.use_tls:
				smtp.starttls()

			if self.user and self.password:
				smtp.login(self.user, self.password)

			smtp.sendmail(self.user, self.recipients, msg.as_string())
			#smtp.send_message(msg)

			log(f"[SMTP] Message sent to {', '.join(self.recipients)}")

		except smtplib.SMTPAuthenticationError as e:
			log(f"[SMTP] Login failed: {str(e)}")

		except smtplib.SMTPRecipientsRefused as e:
			log(f"[SMTP] One or more recipients refused: {str(e)}")

		except smtplib.SMTPException as e:
			log(f"[SMPT] Auth failed due to unsuppprted mechanism: {str(e)}")

		except Exception as e:
			log(f"[SMTP] Send failed with unexpeted error: {str(e)}")

		finally:
			smtp.quit()
			self.reset_attachments()


def id2name(id, classes):
	for key, value in classes.items():
		if id == key:
			return value


def analyze(buffer):
	threshold = 0.6

	img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

	h, w = img.shape[:2]

	blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), swapRB=True, crop=False)
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	objects = []
	# loop over the detections
	for detection in detections[0, 0]:
		classID, score, x1, y1, x2, y2 = detection[1:7]

		# filter out weak detections by ensuring the 'score' is greater than the confidence threshold
		if score > threshold:
			object = id2name(classID, classNames)
			if  object != 'Person':
				continue

			objects.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])
	objects = np.array(objects)

	return img, objects


def HOG_analyze(buffer):
	threshold = 0.6

	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}

	img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

	boxes, weights = hog.detectMultiScale(img, **hogParams)

	objects = []
	for i, (x, y, w, h) in enumerate(boxes):
		if weights[i] > threshold:
			# the HOG detector returns slightly larger rectangles than the real objects.
			# so we slightly shrink the rectangles to get a nicer output.
			pad_w, pad_h = int(0.15 * w), int(0.05 * h)
			objects.append([x + pad_w, y + pad_h, x + w - pad_w, y + h - pad_h])
	objects = np.array(objects)
	objects = non_max_suppression(objects, overlapThresh=0.6)

	return img, objects


def showSnapshot(buffer, name, objects=None, detect=False, resize=None, savedir=None, source=None):
	log(f"{'[' + source+ ']' if source else ''} Opening snapshot window")

	#winname = name + ' (Press any key to exit; Press \'s\' to save the snapshot)'
	winname = name + ' (Press \'q\' to quit, \'s\' to save the snapshot)'
	cv2.namedWindow(winname)
	cv2.startWindowThread()

	if isinstance(buffer, np.ndarray):
		img = buffer
	else:
		if detect: # and objects is None:
			img, objects = analyze(buffer)
		else:
			img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

	try:
		if detect and objetcs is not None:
			#log(f"{'[' + source+ ']' if source else ''} {len(objects)} object{'' if num == 1 else 's'} detected")

			# Draw a rectangle around the objects
			for (x1, y1, x2, y2) in objects:
				cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

		# resize if parameters are given
		#if resize and len(resize) == 2:
		#	img2 = cv2.resize(img, resize)
		#else:
		#	img2 = img

		#cv2.imshow(winname, img2)
		cv2.imshow(winname, cv2.resize(img, resize) if resize and len(resize)==2 else img)
		cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)

		# wait for key pressed
		#key = cv2.waitKey(0) & 0xFF

		# the following is necessary on the mac
		while True:
			key = cv2.waitKey(1) & 0xFF

			if key == ord('s'):
				outFile = os.path.join(savedir or os.getcwd(), name)

				if os.path.isfile(outFile):
					outFile = outFile[:-4] + '_1' + outFile[-4:]

				cv2.imwrite(outFile, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
				log(f"{'[' + source+ ']' if source else ''} Snapshot saved as {outFile}")
				break

			if key == ord('q'):
				break

	except (KeyboardInterrupt, SystemExit):
		pass

	finally:
		if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) > 0:
			cv2.destroyWindow(winname)

		log(f"{'[' + source+ ']' if source else ''} Snapshot window closed")

		# the following is necessary on the mac
		if sys.platform == 'darwin':
			cv2.waitKey(1)


#
# Source:
# https://github.com/pnbruckner/homeassistant-config/blob/3ca2d9db735dfc026ec02f08bb00006a90730b4d/tools/test_events.py
#

class LoginError(Exception):
	"""A login error occcured"""


class EventMonitor():
	def __init__(self, camera, smtpconf=None, snapconf=None):
		if camera['timeout']:
			self.SNAPMGR_URL = 'http://{host}:{port}/cgi-bin/snapManager.cgi?action=attachFileProc&Flags[0]=Event&Events=[{events}]&heartbeat={timeout}'
		else:
			self.SNAPMGR_URL = 'http://{host}:{port}/cgi-bin/snapManager.cgi?action=attachFileProc&Flags[0]=Event&Events=[{events}]'

		#self.GETTIME_URL = 'http://{host}:{port}/cgi-bin/global.cgi?action=getCurrentTime'

		self.camera = camera
		self.connected = False

		#self.URL = self.EVENTMGR_URL.format(**camera)
		self.URL = self.SNAPMGR_URL.format(**camera)

		self.process = None
		self.timeout = int(camera['timeout']) + 5 if camera['timeout'] else None

		self.snapname = ''

		self.detected = 0

		self.smtpconf = smtpconf

		self.snapconf = {
			'detect':	False,
			'resize':	None,
			'savedir':	None
			}

		self.show = snapconf['show'] if isinstance(snapconf, dict) and 'show' in snapconf else False
		if isinstance(snapconf, dict):
			for key in self.snapconf.keys():
				if key in snapconf:
					self.snapconf[key] = snapconf[key]
		self.snapconf['source'] = camera['name']


	def onConnect(self):
		log(f"[{self.camera['name']}] Connected to snapshot event manager on {self.camera['host']}")
		self.connected = True


	def onDisconnect(self, reason):
		log(f"[{self.camera['name']}] Disconnected from snapshot event manager on {self.camera['host']}. Reason: {reason}")
		self.connected = False


	def onEvent(self, event, data):
		log(f"[{self.camera['name']}] Event: {event['code']} {event['action']} @ {event['region']}")

		if event['code'] not in self.camera['events'] and self.camera['events'] != 'All':
			return

		if event['index'] == 1:
			self.snapname = f"{self.camera['name'].replace(' ', '')}_{event['timestamp'].replace('-', '').replace(':', '').replace(' ', '')}"
		name = self.snapname + f"{'_' + str(event['index']) + 'of' + str(event['count']) if event['count'] > 1 else ''}.jpg"

		image, objects = analyze(data)

		if self.snapconf['savedir']:
			outFile = os.path.join(self.snapconf['savedir'], name)

			try:
				with open(outFile, 'wb') as file:
					file.write(data)
				log(f"[{self.camera['name']}] Snapshot file saved as {outFile}")
			except Exception as e:
				log(f"[{self.camera['name']}] Saving snapshot as {outFile} failed: {str(e)}")

		if not self.snapconf['detect'] or len(objects) > 0:
			if self.show:
				Process(target=showSnapshot, args=(image, name, objects, ), kwargs=self.snapconf).start()
		else:
			log(f"[{self.camera['name']}] No person detected")

		if self.smtpconf and self.smtpconf['mailer']:
			self.detected = max(len(objects), self.detected)

			try:
				self.smtpconf['mailer'].attach(data, name)

				if event['index'] == event['count']:
					if not self.snapconf['detect'] or self.detected > 0:
						log(f"[{self.camera['name']}] Sending message ({self.detected} person{'' if self.detected == 1 else 's'} detected) ...")
						self.smtpconf['mailer'].send(self.smtpconf['subject'].format(**event), self.smtpconf['body'].format(**event))
						self.detected = 0
					else:
						log(f"[{self.camera['name']}] Message cancelled")
						self.smtpconf['mailer'].reset_attachments()

			except KeyError as e:
				log(f"[{self.camera['name']}] Key Error: {str(e)} not defined")


	def connect(self, retries=0):
		response = None

		with requests.Session() as session:
			if 'user' in self.camera and 'password' in self.camera: #and self.camera['user']:
				if 'auth' in self.camera:
					if self.camera['auth'] == 'digest':
						session.auth = HTTPDigestAuth(self.camera['user'], self.camera['password'])
					else:
						session.auth = HTTPBasicAuth(self.camera['user'], self.camera['password'])

			for i in range(1, 2 + retries):
				if i > 1:
					log(f"[{self.camera['name']}] Retrying ...")
				try:
					#response = session.get(self.URL, timeout=(3.05, None), stream=True, verify=True)
					response = session.get(self.URL, timeout=(3.05, self.timeout), stream=True, verify=True)
					if response.status_code == 401:
						raise LoginError
					response.raise_for_status()
				except LoginError:
					log(f"[{self.camera['name']}] Login error! Check username and password")
					break
				except (RequestException, ConnectionError) as e:
					#log(f"[{self.camera['name']}] Failed to retrieve data: {str(e)}")
					log(f"[{self.camera['name']}] Failed to retrieve data") #No Response?
					continue
				else:
					break

		return response


	def _bytes(self, response):
		content_type = response.headers['content-type']

		index = content_type.rfind('boundary=')
		assert index != 1

		encoding = response.encoding or 'utf-8'

		boundary = content_type[index + len('boundary='):] + '\r\n'
		boundary = boundary.encode(encoding)

		rd = BufferedReader(response.raw, buffer_size=10)

		while True:
			self._skip_to_boundary(rd, boundary)
			type, length = self._parse_length(rd, encoding)

			if type == 'text/plain':
				event = {
					'source':	self.camera['name'],
					'timestamp': datetime.now().replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S'),
					'action':	None,
					'code':		None,
					'region':	None,
					'count':	1,
					'index':	1
					}

				region = []

				for line in rd.read(length).decode(encoding).split('\r\n'):
					#log(f"[{self.camera['name']}] Received: {line}")
					if line.startswith('Events[0].EventBaseInfo.Action'):
						event['action'] = line.split('=')[1]
					elif line.startswith('Events[0].EventBaseInfo.Code'):
						event['code'] = line.split('=')[1]
					elif line.startswith('Events[0].CountInGroup'):
						event['count'] = int(line.split('=')[1])
					elif line.startswith('Events[0].IndexInGroup'):
						event['index'] = int(line.split('=')[1])
					elif line.startswith('Events[0].RegionName'):
						region.append(line.split('=')[1])
					#elif line.startswith('Heartbeat'):
					##else:
					#	log(f"[{self.camera['name']}] Received: {line}")

				event['region'] = ', '.join(region)

			if type == 'image/jpeg': # and length > 0:
				yield event, rd.read(length)


	def _parse_length(self, rd, encoding):
		length = 0
		type = None

		while True:
			line = rd.readline()
			if line == b'\r\n':
				return type, length
			if line.startswith(b'Content-Type'):
				type = line.decode(encoding).split(':')[1].strip()
			if line.startswith(b'Content-Length'):
				length = int(line.decode(encoding).split(':')[1].strip())
				assert length > 0


	def _skip_to_boundary(self, rd, boundary: bytes):
		for _ in range(10):
			if boundary in rd.readline():
				break
		else:
			raise RuntimeError('Boundary not detected:', boundary)


	def start(self):
		self.process = Process(target=self._start)
		self.process.start()

		return self.process


	def _start(self):
		reason = "Unknown"

		while True:
			response = self.connect(retries=1)

			if response and response.status_code == 200:
				if not response.encoding:
					response.encoding = 'utf-8'

				try:
					self.onConnect()

					for event, data in self._bytes(response):
						self.onEvent(event, data)

					reason = "No response"
				except (KeyboardInterrupt, SystemExit):
					reason = "User terminated"
					break
				except Exception as e:
					reason = str(e)
					if not 'Read timed out.' in reason:
						break
				finally:
					self.onDisconnect(reason)
					response.close()
			else:
				log(f"[{self.camera['name']}] Unable to connect")
				break


def myexcepthook(exctype, value, traceback):
	pass
	#log(f"Value: {exctype}")
	#for p in active_children():
	#	p.terminate()


# the following is necessary on the mac
if sys.platform == 'darwin':
	sys.excepthook = myexcepthook


if __name__ == '__main__':
	log("Starting Snapshot Event Monitor ...")

	snapParms = {}
	smtpParms = {
		'mailer':	None,
		'subject':	'{source} has captured an event',
		'body': 	'{source} has captured an event of type {code} in region {region} at {timestamp}'
		}

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--config", type=str, default="camera.cfg",
		help="name or path of the config file (default: camera.cfg)")
	ap.add_argument("-d", "--detect", action="store_true",
		help="enable people detection (default: False)")
	ap.add_argument("-o", "--out", type=str, default=None,
		help="output directory for the saved snapshots (default: No saving) ")
	ap.add_argument("-r", "--resize", type=str, default=None,
		help="resize output format: WidthxHeight (default: No resizing)")
	ap.add_argument("-s", "--show", action="store_true",
		help="show snapshot of event (default: False)")
	ap.add_argument("-t", "--timeout", type=int, default=0,
		help="heartbeat/reply timeout in seconds (default: 0=no timeout)")
	ap.add_argument("-q", "--quiet", action="store_true",
		help="suppresses console output (default: False)")
	args = vars(ap.parse_args())

	snapParms['show']    = args['show']
	snapParms['detect']  = args['detect']
	snapParms['savedir'] = os.getcwd() if args['out'] is not None and not os.path.isdir(args['out']) else args['out']

	try:
		snapParms['resize'] = tuple([int(x) for x in args['resize'].lower().split('x')])
	except:
		snapParms['resize'] = None #(640, 360)

	configFile = os.path.join(os.getcwd(), configFile) if not os.path.isfile(args['config']) else args['config']
	if not os.path.isfile(configFile):
		log("No config file! Exit")
		sys.exit(1)

	if args['quiet']:
		devnull = open(os.devnull, 'w')
		sys.stdout = devnull

	config = configparser.ConfigParser()
	config.read([configFile], encoding='utf-8')

	camList = []

	log(f"Reading config from {configFile} ...")
	for section in config.sections():
		try:
			if section == 'SMTP':
				smtp = {
					'server':	config.get(section, 'server'),
					'port':		int(config.get(section, 'port', fallback='587')),
					'use_tls':	bool(config.get(section, 'tls', fallback='True')),
					'sender':	config.get(section, 'sender', fallback='Snapshot Event Monitor'),
					'user':		config.get(section, 'user', fallback=''),
					'password':	config.get(section, 'password', fallback='')
					}

				smtpParms['subject'] = config.get(section, 'subject', fallback=smtpParms['subject'] )
				smtpParms['body']    = config.get(section, 'message', fallback=smtpParms['body'] )

				toList = config.get(section, 'recipients')

				log(f"[{section}] Config okay. Initializing ...")

				smtpParms['mailer'] = Mailer(**smtp)
				for recipient in toList.split(','):
					smtpParms['mailer'].add_recipient(recipient.strip())

			else:
				camera = {
					'name':		section,
					'host':		config.get(section, 'host'),
					'port':		int(config.get(section, 'port', fallback='80')),
					'user':		config.get(section, 'user'),
					'password':	config.get(section, 'password'),
					'auth': 		config.get(section, 'auth', fallback='digest'),
					'events': 	config.get(section, 'events', fallback='VideoMotion'),
					'timeout':	args['timeout'] or None
					}

				log(f"[{section}] Config okay")

				camList.append(camera)

		except:
			log(f"[{section}] Error! Check your configuration")

	procList = []

	for cam in camList:
		proc = EventMonitor(cam, smtpconf=smtpParms, snapconf=snapParms).start()
		procList.append(proc)

	for proc in procList:
		proc.join()
