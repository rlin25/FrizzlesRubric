                          +---------------------+
                          |     Internet        |
                          +----------+----------+
                                     |
                            Internet Gateway (IGW)
                                     |
                           +--------------------+
                           |    Public Subnet    |  (e.g. subnet-xxxxxxxx)
                           |  CIDR: 172.31.49.0/24|
                           +---------+----------+
                                     |
          +--------------------------+-------------------------+
          |                          |                         |
+---------v---------+      +---------v----------+    +---------v----------+
|  Bastion Host     |      |  NAT Gateway       |    |  Elastic IP (EIP)  |
| (Public Subnet)   |      |  (in Public Subnet)|    |  assigned to NAT   |
| Public IPv4 IP    |      | Public IP via EIP  |    | Gateway            |
+-------------------+      +--------------------+    +--------------------+
         |                                   |
         |                                   |
         |                                   |
         |          Route Table (Public)     |
         |          0.0.0.0/0 → IGW          |
         +-----------------------------------+

                           +--------------------+
                           |   Private Subnet    | (e.g. subnet-xxxxxxxx)
                           |  CIDR: 172.31.20.0/24|
                           +---------+----------+
                                     |
                             Private Instances
                        (No Public IP, no IGW route)
                                     |
                Route Table (Private) 0.0.0.0/0 → NAT Gateway

---

### Key Points:

- **Public subnet** has route `0.0.0.0/0 → IGW`, hosts:
  - Bastion Host (with public IP for SSH)
  - NAT Gateway (with Elastic IP to allow outbound internet)
  
- **Private subnet** has route `0.0.0.0/0 → NAT Gateway`, hosts:
  - Private EC2 instances without public IPs
  
- **NAT Gateway** enables private instances to access the internet securely without exposing them publicly.

- Bastion host allows SSH access into private instances via jump server pattern.