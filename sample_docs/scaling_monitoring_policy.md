# Scaling and Monitoring Policy

Fictional teaching document for the CloudRAG project. These rules describe an imaginary university cloud lab, not a real institution or the implemented CloudRAG application.

## Capacity rules

The API service scales out by adding one replica when average CPU utilization remains above 70 percent for 5 consecutive minutes. It scales in by removing one replica when average CPU utilization stays below 30 percent for 15 consecutive minutes. The service keeps at least 1 replica and at most 4 replicas. A replica is another running copy of the same service.

The API replicas share a separate database and document store; they do not keep the only copy of documents on their own local disks. The model service is scaled separately because its memory and compute needs differ from the API service. A queue handles document ingestion so that large uploads do not block interactive questions.

## Monitoring and alerts

The dashboard monitors request count, error rate, p95 response latency, CPU utilization, memory usage, disk usage, queue length, and model availability. The p95 response latency is the duration within which 95 percent of measured requests finish.

An alert is raised when the API error rate exceeds 5 percent for 5 minutes or when p95 response latency exceeds 2 seconds for 10 minutes. These are sample alert thresholds, not demonstrated CloudRAG performance results. Logs record request identifiers and error categories while avoiding full document text and private question content. The team reviews capacity and resource costs weekly.
