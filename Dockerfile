FROM nlpbox/nlpbox-base:16.04

RUN apt-get update && apt-get install python-pip -y

WORKDIR /opt
RUN wget http://hal3.name/megam/megam_src.tgz && \
    tar xzf megam_src.tgz && rm megam_src.tgz && \
    git clone https://github.com/arne-cl/nltk-maxent-pos-tagger.git

WORKDIR /opt/megam_0.92
# we can't compile without this symlink,
# cf. http://stackoverflow.com/questions/13584629/ocaml-compile-error-usr-bin-ld-cannot-find-lstr

RUN apt-get install ocaml -y && \
    ln -s /usr/lib/ocaml/libcamlstr.a /usr/lib/ocaml/libstr.a && \
    make opt && \
    mv megam.opt /usr/bin/megam && \
    apt-get purge ocaml -y && apt autoremove -y

WORKDIR /opt/nltk-maxent-pos-tagger
RUN pip install -r requirements.txt && \
    python -c "import nltk; nltk.download('brown'); nltk.download('treebank')"
