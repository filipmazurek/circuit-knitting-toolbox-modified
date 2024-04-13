from azure.quantum.qiskit.job import AzureQuantumJob
from azure.quantum.qiskit import AzureQuantumProvider
from azure.quantum import Workspace

# Create code so that we can get job results from Azure Quantum based on job id's
workspace = Workspace(
            resource_id = "/subscriptions/4ed1f7fd-7d9e-4c61-9fac-521649937e65/resourceGroups/Cutting/providers/Microsoft.Quantum/Workspaces/Cutting",
            location = "eastus")


provider = AzureQuantumProvider(workspace)

job = provider.get_job('7ff8a3c8-d677-11ee-861c-acde48001122')

print(job.result().results[0].data)
