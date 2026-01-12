import 'package:flutter/material.dart';

class ApplicantsListPage extends StatefulWidget {
  const ApplicantsListPage({super.key});

  @override
  State<ApplicantsListPage> createState() => _ApplicantsListPageState();
}

class _ApplicantsListPageState extends State<ApplicantsListPage> {
  String _selectedStatus = 'All';

  final List<Map<String, dynamic>> _mockApplicants = [
    {
      'id': '1',
      'name': 'Alex Tan',
      'jobTitle': 'Senior Flutter Developer',
      'status': 'New',
      'appliedDate': '2 hours ago',
    },
    {
      'id': '2',
      'name': 'Maya Lee',
      'jobTitle': 'Senior Flutter Developer',
      'status': 'Shortlisted',
      'appliedDate': '1 day ago',
    },
    {
      'id': '3',
      'name': 'John Doe',
      'jobTitle': 'FastAPI Backend Engineer',
      'status': 'New',
      'appliedDate': '3 hours ago',
    },
  ];

  @override
  Widget build(BuildContext context) {
    final filteredApplicants = _selectedStatus == 'All'
        ? _mockApplicants
        : _mockApplicants.where((a) => a['status'] == _selectedStatus).toList();

    return Column(
      children: [
        // Title
        Padding(
          padding: const EdgeInsets.all(16),
          child: const Row(
            children: [
              Text(
                'Applicants',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
              ),
            ],
          ),
        ),
        // Filter
        Container(
          color: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: ['All', 'New', 'Shortlisted', 'Rejected']
                  .map((status) => Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: FilterChip(
                          label: Text(status),
                          selected: _selectedStatus == status,
                          onSelected: (selected) =>
                              setState(() => _selectedStatus = status),
                        ),
                      ))
                  .toList(),
            ),
          ),
        ),
        const Divider(height: 1),
        // List
        Expanded(
          child: filteredApplicants.isEmpty
              ? const Center(
                  child: Text('No applicants'),
                )
              : ListView.separated(
                  padding: const EdgeInsets.all(16),
                  itemCount: filteredApplicants.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final applicant = filteredApplicants[index];
                    return Container(
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(color: const Color(0xFFE2E8F0)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              CircleAvatar(
                                radius: 20,
                                backgroundColor: const Color(0xFFEEF2FF),
                                child: Text(
                                  applicant['name']
                                      .toString()
                                      .split(' ')[0][0]
                                      .toUpperCase(),
                                  style:
                                      const TextStyle(color: Color(0xFF4F46E5)),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      applicant['name'],
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      applicant['jobTitle'],
                                      style: const TextStyle(
                                        fontSize: 12,
                                        color: Colors.black54,
                                      ),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}
