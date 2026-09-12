# Backup and Recovery Policy

Fictional teaching document for the CloudRAG project. These rules describe an imaginary university cloud lab, not a real institution or the implemented CloudRAG application.

## Backup schedule

Incremental backups are taken every day at 01:00 UTC. A full backup is taken every Sunday at 02:00 UTC. The Sunday full backup is additional to the daily incremental backup. An incremental backup saves changes since the previous backup; a full backup saves the complete selected dataset.

The lab backs up its course documents, application configuration, and database. Backup sets are retained for 28 days. A backup set includes the full backup and the incremental files needed to restore it. The operations team checks job completion every morning and investigates failed jobs before the next scheduled backup.

## Recovery targets

The lab's recovery point objective (RPO) is 24 hours: the design aims to lose no more than 24 hours of changes after an incident. The recovery time objective (RTO) is 4 hours: the design aims to restore service within 4 hours. These are planning targets, not measured or guaranteed achievements. Meeting them assumes successful backups, available restore infrastructure, and a usable copy outside the failed system.

The team performs a restore exercise monthly and records the actual recovery duration and latest recovered timestamp. Backups are encrypted and restricted to the operations team. A second backup copy is stored separately from the primary application host.
