

window.onload = function() {
    localStorage.setItem("file", $('#file').val());
}

var file = localStorage.getItem('file');
document.querySelector('#file') = file;