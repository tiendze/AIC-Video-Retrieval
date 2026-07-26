# Multi-stage build adapted from the plugin's own reference Dockerfile
# (https://github.com/duydo/elasticsearch-analysis-vietnamese/blob/v9.3.0/Dockerfile).
# references.md §4.3 only documents `elasticsearch-plugin install file://...zip`,
# which is not enough on its own: the plugin wraps CocCoc's C++ tokenizer via
# JNI, so a native .so must also be compiled and placed on java.library.path,
# or Elasticsearch crashes at startup with UnsatisfiedLinkError.
ARG ES_VERSION=9.3.0

# Stage 1: build the native coccoc-tokenizer JNI library (C++/cmake).
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION} AS coccoc-builder

USER root
ENV JAVA_HOME=/usr/share/elasticsearch/jdk
ENV PATH=$JAVA_HOME/bin:$PATH

RUN microdnf install -y --nodocs \
        gcc gcc-c++ make cmake pkgconf-pkg-config git gzip tar which \
    && microdnf clean all

WORKDIR /tmp
RUN git clone --depth 1 https://github.com/duydo/coccoc-tokenizer.git \
    && mkdir coccoc-tokenizer/build

WORKDIR /tmp/coccoc-tokenizer/build
RUN cmake -DJAVA_HOME=${JAVA_HOME} -DBUILD_JAVA=1 .. && make install

# Stage 2: build the elasticsearch-analysis-vietnamese plugin jar.
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION} AS plugin-builder

USER root
ENV JAVA_HOME=/usr/share/elasticsearch/jdk
ENV PATH=$JAVA_HOME/bin:$PATH

ARG MVN_VERSION=3.9.9
# curl is intentionally not installed here: the base image already ships
# curl-minimal, and requesting the full `curl` package conflicts with it.
RUN microdnf install -y --nodocs gzip tar \
    && microdnf clean all \
    && curl -fsSL -o /tmp/maven.tar.gz \
        https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/${MVN_VERSION}/apache-maven-${MVN_VERSION}-bin.tar.gz \
    && tar xf /tmp/maven.tar.gz -C /opt \
    && rm /tmp/maven.tar.gz
ENV MVN_HOME=/opt/apache-maven-${MVN_VERSION}
ENV PATH=$MVN_HOME/bin:$PATH

WORKDIR /tmp/elasticsearch-analysis-vietnamese

# Pinned to the v${ES_VERSION} tag, matching the plugin/Elasticsearch
# compatibility table in references.md §4.3 (plugin 9.3.0 <-> ES 9.0.0-9.3.0).
ARG ES_VERSION
RUN curl -fsSL -o source.tar.gz \
        https://github.com/duydo/elasticsearch-analysis-vietnamese/archive/refs/tags/v${ES_VERSION}.tar.gz \
    && tar xzf source.tar.gz --strip-components=1 \
    && rm source.tar.gz

RUN mvn --batch-mode -Dmaven.test.skip -e package -DprojectVersion=$ES_VERSION

# Stage 3: final image — combine the native lib + plugin jar into Elasticsearch.
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}

ARG ES_VERSION
ARG COCCOC_INSTALL_PATH=/usr/local
ARG COCCOC_DICT_PATH=$COCCOC_INSTALL_PATH/share/tokenizer/dicts

USER root
COPY --from=coccoc-builder $COCCOC_INSTALL_PATH/lib/libcoccoc_tokenizer_jni.so /usr/lib/
COPY --from=coccoc-builder $COCCOC_DICT_PATH $COCCOC_DICT_PATH
COPY --from=plugin-builder \
    /tmp/elasticsearch-analysis-vietnamese/target/releases/elasticsearch-analysis-vietnamese-$ES_VERSION.zip /tmp/
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch \
        file:///tmp/elasticsearch-analysis-vietnamese-$ES_VERSION.zip \
    && rm /tmp/elasticsearch-analysis-vietnamese-$ES_VERSION.zip

USER elasticsearch
