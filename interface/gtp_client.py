import argparse
import traceback
import sys

import urllib.request
import urllib.parse


parser = argparse.ArgumentParser(description="Client for RocAlphaGo GTP Engine")
parser.add_argument("--host", type=str, default="192.168.88.238", help="GTP engine host")
parser.add_argument("--port", "-p", type=int, default=5000, help="GTP engine port")
args = parser.parse_args()

sys.stderr.write("GTP engine ready\n")
sys.stderr.flush()

while True:
    try:
        inpt = input()

        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]

        for cmd in cmd_list:
            cmd = urllib.parse.quote(cmd)
            response = urllib.request.urlopen("http://{host}:{port}/gtp?cmd={cmd}".format(host=args.host, port=args.port, cmd=cmd))
            engine_reply = response.read().decode("utf8")
            sys.stdout.write(engine_reply)
    except KeyboardInterrupt:
        break
    except EOFError:
        break
    except:
        err, msg, _ = sys.exc_info()
        sys.stderr.write("{} {}\n".format(err, msg))
        sys.stdout.write("? system error")