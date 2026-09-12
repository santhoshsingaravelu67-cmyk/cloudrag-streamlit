# Access and Network Policy

Fictional teaching document for the CloudRAG project. These rules describe an imaginary university cloud lab, not a real institution or the implemented CloudRAG application.

## Identity and permissions

All administrators must use multi-factor authentication (MFA). Each administrator has an individual account; shared administrator credentials are prohibited. Students receive read-only access to their assigned course resources. Teaching assistants may upload approved course documents. Only the operations team may change infrastructure settings or backup permissions.

Permissions follow least privilege: each account receives only the access needed for its role. The operations team reviews access every month and removes access when a person leaves the course. Service credentials are stored in a secrets manager and are never committed to source code.

## Network boundaries

The database runs in a private network and is not directly accessible from the public internet. Only the application backend may connect to the database. User requests enter through an HTTPS gateway on port 443. The backend validates the user's identity and document permissions before returning any retrieved content.

Administrative SSH access requires the private VPN. Public SSH access is disabled. Firewall rules allow only the required source, destination, and port combinations. Network logs record denied connections for investigation. Uploaded documents are treated as data; their text must not override application instructions or grant additional permissions.
