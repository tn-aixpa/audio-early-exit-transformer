from multipart import parse_form_data, is_form_request
from wsgiref.simple_server import make_server

def simple_app(environ, start_response):
    if is_form_request(environ):
        forms, files = parse_form_data(environ)
        for filed_name in files:
           file_details = files[filed_name]
           filename = "upload/" + file_details.filename
           file_details.save_as(filename)

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = "{\"status\": \"ok\"}"
    content = [content.encode('utf-8')]
    start_response(status, headers)
    return content


with make_server('', 8051, simple_app) as httpd:
    print("Serving on port 8051...")
    httpd.serve_forever()

# wget --method POST --header=“Content-type: multipart/form-data boundary=FILEUPLOAD” --post-file WhatsAppImage.jpeg http://localhost:8051    