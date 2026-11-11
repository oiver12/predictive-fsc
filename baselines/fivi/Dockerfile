# Build from the repository root after building the core image:
#   docker build -t predictive-fsc .
#   docker build --platform linux/amd64 -t predictive-fsc-fivi baselines/fivi
ARG CORE_IMAGE=predictive-fsc
ARG MAVEN_IMAGE=maven:3.9.9-eclipse-temurin-21@sha256:3a4ab3276a087bf276f79cae96b1af04f53731bec53fb2e651aca79e4b10211e

FROM ${MAVEN_IMAGE} AS build
ENV MAVEN_OPTS="-Xmx512m -XX:ActiveProcessorCount=1"
ADD --checksum=sha256:322d3a51babb9d68a87a5f02696315ff2729f796bba82d2cfdc7ae459ece136e \
    https://codeload.github.com/oiver12/ConstrainedPlanningToolbox/tar.gz/d97f25187803f52b46ebcf040bf9fb8b5e832ed2 /tmp/fivi.tar.gz
# The bundled lpsolve JNI library is x86_64 only.
RUN test "$(uname -m)" = x86_64
RUN mkdir -p /opt/fivi/source && tar -xzf /tmp/fivi.tar.gz --strip-components=1 -C /opt/fivi/source
RUN mvn -B -q -f /opt/fivi/source/pom.xml package

FROM ${CORE_IMAGE}
USER root
COPY --from=build /opt/java/openjdk /opt/java/openjdk
COPY --from=build /opt/fivi/source /opt/fivi/source
COPY --from=build /opt/fivi/source/target/FiVI.jar /opt/fivi/fivi.jar
COPY --from=build /opt/fivi/source/server/linux/lpsolve_64 /opt/fivi/native
COPY build_info.py /tmp/fivi-build-info.py
COPY fivi /usr/local/bin/fivi
ENV PATH=/opt/java/openjdk/bin:$PATH LD_LIBRARY_PATH=/opt/fivi/native
RUN chmod +x /usr/local/bin/fivi && python /tmp/fivi-build-info.py
USER runner
ENTRYPOINT ["fivi"]
CMD ["--help"]
