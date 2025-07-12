#!/usr/bin/ruby
############################################################
# ⓒ  Tzolkin Garduño, Gunnar Wolf 2025
############################################################
# MIT-like licensing.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
############################################################
require 'sinatra'
require 'socket'

template :layout do
  File.open('views/layout.html').read()
end

get '/' do
  return erb(File.open('views/index.erb').read())
end

post '/query' do
  begin
    return erb(get_results(params))
  rescue Errno::ECONNREFUSED
    status 500
    return '<h3>Cannot connect to language model server</h3><p>Gateway error</p>'
  end
end

# %w(deb_logo.png debhome.css rag.css openlogo-50.png).each do |file|
#   get '/' + file do
#     send_file(File.join('./static', file))
#   end
# end

get('/openlogo-50.png') { send_file './static/openlogo-50.png' }
get('/favicon.ico')     { send_file './static/favicon.ico' }
get('/debhome.css')     { send_file './static/debhome.css' }
get('/debian.css')      { send_file './static/debian.css' }
get('/rag.css')         { send_file './static/rag.css' }

def get_results(params)
  res = "<p>Querying for: <em>#{ params[:query] }</em></p>"

  socket = TCPSocket.new('localhost', 31337) or
    raise RuntimeError, 'Cannot connect'

  socket.puts(params[:query])
  while line = socket.gets
    res << line
  end
  socket.close

  return res
end
