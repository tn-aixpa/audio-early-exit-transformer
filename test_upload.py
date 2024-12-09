import multipart
from wsgiref.simple_server import make_server

def simple_app(environ, start_response):
    fields = {}
    files = {}
    def on_field(field):
        fields[field.field_name] = field.value
    def on_file(file):
        files[file.field_name] = {'name': file.file_name.decode("utf-8"), 'file_object': file.file_object}

    #'PATH_INFO' 'REQUEST_METHOD'
    if environ['PATH_INFO'] == '/upload':
        multipart_headers = {'Content-Type': environ['CONTENT_TYPE']}
        multipart_headers['Content-Length'] = environ['CONTENT_LENGTH']
        multipart.parse_form(multipart_headers, environ['wsgi.input'], on_field, on_file)
        for filed_name, each_file_details in files.items():
            filename = "upload/" + each_file_details['name']
            with open(filename, 'wb') as f:
                uploaded_file = each_file_details['file_object']
                uploaded_file.seek(0)
                f.write(uploaded_file.read())

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = "{\"status\": \"ok\"}"
    content = [content.encode('utf-8')]
    start_response(status, headers)
    return content

with make_server('', 8051, simple_app) as httpd:
    print("Serving on port 8051...")
    httpd.serve_forever()