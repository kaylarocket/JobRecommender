import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/form_input.dart';
import '../../widgets/primary_button.dart';
import '../recruiter/recruiter_dashboard_page.dart';
import '../seeker/seeker_shell.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key, this.initialRole = 'job_seeker'});
  final String initialRole;

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final fullNameCtrl = TextEditingController();
  final emailCtrl = TextEditingController();
  final passwordCtrl = TextEditingController();
  final confirmCtrl = TextEditingController();
  final locationCtrl = TextEditingController();
  final headlineCtrl = TextEditingController();
  final skillsCtrl = TextEditingController();
  final yearsCtrl = TextEditingController();
  late String selectedRole;

  @override
  void initState() {
    super.initState();
    selectedRole = widget.initialRole;
  }

  @override
  void dispose() {
    fullNameCtrl.dispose();
    emailCtrl.dispose();
    passwordCtrl.dispose();
    confirmCtrl.dispose();
    locationCtrl.dispose();
    headlineCtrl.dispose();
    skillsCtrl.dispose();
    yearsCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    return Scaffold(
      appBar: AppBar(title: const Text('Create account')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                ChoiceChip(
                  label: const Text('Job Seeker'),
                  selected: selectedRole == 'job_seeker',
                  onSelected: (_) => setState(() => selectedRole = 'job_seeker'),
                ),
                const SizedBox(width: 10),
                ChoiceChip(
                  label: const Text('Recruiter'),
                  selected: selectedRole == 'recruiter',
                  onSelected: (_) => setState(() => selectedRole = 'recruiter'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            FormInput(label: 'Full name', controller: fullNameCtrl, hint: 'Alex Doe'),
            const SizedBox(height: 12),
            FormInput(label: 'Email', controller: emailCtrl, hint: 'alex@example.com', keyboardType: TextInputType.emailAddress),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: FormInput(label: 'Password', controller: passwordCtrl, obscure: true, hint: '••••••••')),
              const SizedBox(width: 12),
              Expanded(child: FormInput(label: 'Confirm', controller: confirmCtrl, obscure: true, hint: '••••••••')),
            ]),
            const SizedBox(height: 12),
            FormInput(label: 'Preferred location (Country/City)', controller: locationCtrl, hint: 'Kuala Lumpur'),
            const SizedBox(height: 12),
            FormInput(label: 'Headline / target role', controller: headlineCtrl, hint: 'Frontend Engineer'),
            const SizedBox(height: 12),
            FormInput(label: 'Key skills', controller: skillsCtrl, hint: 'React, Flutter, APIs'),
            const SizedBox(height: 12),
            FormInput(label: 'Years of experience', controller: yearsCtrl, hint: '4', keyboardType: TextInputType.number),
            const SizedBox(height: 20),
            PrimaryButton(
              label: 'Create account',
              loading: auth.isLoading,
              onPressed: () async {
                if (passwordCtrl.text != confirmCtrl.text) {
                  _showError(context, 'Passwords do not match');
                  return;
                }
                await auth.register(
                  fullName: fullNameCtrl.text,
                  email: emailCtrl.text.trim(),
                  password: passwordCtrl.text,
                  role: selectedRole,
                  preferredLocation: locationCtrl.text,
                  headline: headlineCtrl.text,
                  skills: skillsCtrl.text,
                  experienceYears: int.tryParse(yearsCtrl.text),
                );
                if (auth.session != null) {
                  _goToDashboard(context, auth);
                  context.read<JobProvider>().loadJobs();
                } else if (auth.error != null) {
                  _showError(context, auth.error!);
                }
              },
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Back to sign in', style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w700)),
            )
          ],
        ),
      ),
    );
  }

  void _goToDashboard(BuildContext context, AuthProvider auth) {
    final role = auth.session?.profile.role;
    if (role == 'recruiter') {
      Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (_) => const RecruiterDashboardPage()), (_) => false);
    } else {
      Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (_) => const SeekerShell()), (_) => false);
    }
  }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }
}
