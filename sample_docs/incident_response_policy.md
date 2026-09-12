# Incident Response Policy

Fictional teaching document for the CloudRAG project. These rules describe an imaginary university cloud lab, not a real institution or the implemented CloudRAG application.

## Ownership and severity

The on-call operations engineer acts as incident lead. The incident lead coordinates investigation, records a timeline, and shares progress with the course coordinator. A complete outage affecting course access for more than 15 minutes is a severity-one incident. A single failed document upload with the rest of the service working is a lower-severity incident.

## Suspected credential compromise

For a suspected credential compromise, the first action is to revoke or disable the affected credential. The team then restricts the affected account, preserves relevant logs, determines the scope of access, and rotates related credentials where necessary. Investigators do not delete logs to hide errors. Access is restored only after the incident lead confirms that the cause has been addressed.

## Recovery and learning

For a service outage, the team checks application health, database connectivity, disk capacity, and model availability. If a recent release caused the failure, the team rolls back to the last known working version. Restore operations use the documented backup procedure.

After recovery, the team checks that questions can retrieve the intended documents and that unauthorized access is still blocked. A review within two working days records the cause, user impact, recovery evidence, and follow-up actions. A completed review does not by itself prove that recovery targets were met.
